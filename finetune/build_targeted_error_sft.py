import json
from pathlib import Path


# ============================================================
# SENTINELAI - TARGETED ERROR SFT DATASET BUILDER
#
# Reads the HUMAN-GROUND-TRUTH evaluation results and extracts
# only windows where the existing LoRA model was wrong.
#
# IMPORTANT:
#   ground_truth = HUMAN label
#   prediction    = existing LoRA prediction
#
# This does NOT modify the human labels.
# This does NOT retrain the model.
# ============================================================


PROJECT_ROOT = Path(r"C:\SentinelAI_Qwen")

EVALUATION_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "test_evaluation"
    / "human_temporal_evaluation.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "sft"
    / "targeted_error_sft.json"
)


CLASSES = [
    "Normal",
    "Fire",
    "Fight",
    "Road Accident",
]


PROMPT = (
    "Analyze this surveillance video clip. "
    "Classify the scene as exactly one of: "
    "Normal, Fire, Fight, or Road Accident. "
    "Return a JSON object with the fields "
    "classification, evidence, and incident_summary."
)


def make_evidence(label):
    if label == "Normal":
        return (
            "No Fire, Fight, or Road Accident is visibly present "
            "in this temporal window."
        )

    if label == "Fire":
        return (
            "Visible fire or fire-related activity is present "
            "in this temporal window."
        )

    if label == "Fight":
        return (
            "A physical fight or altercation is visibly present "
            "in this temporal window."
        )

    if label == "Road Accident":
        return (
            "A road accident or accident-related event is visibly "
            "present in this temporal window."
        )

    return "The scene contains the classified event."


def make_summary(label):
    if label == "Normal":
        return (
            "The video segment shows normal activity without "
            "a targeted security incident."
        )

    if label == "Fire":
        return (
            "The video segment visibly contains a fire-related "
            "incident."
        )

    if label == "Fight":
        return (
            "The video segment visibly contains a physical "
            "altercation."
        )

    if label == "Road Accident":
        return (
            "The video segment visibly contains a road "
            "accident-related event."
        )

    return "The video segment contains the classified event."


def main():

    print()
    print("=" * 70)
    print("SENTINELAI - TARGETED ERROR SFT DATASET")
    print("=" * 70)

    print()
    print("Evaluation:")
    print(EVALUATION_FILE)

    print()
    print("Output:")
    print(OUTPUT_FILE)

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found:\n{EVALUATION_FILE}"
        )

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        evaluation = json.load(f)

    results = evaluation.get("results")

    if not isinstance(results, list):
        raise ValueError(
            "The evaluation file does not contain a valid "
            "'results' list."
        )

    wrong = []

    for item in results:

        ground_truth = item.get(
            "ground_truth"
        )

        prediction = item.get(
            "prediction"
        )

        # Only actual model mistakes.
        if (
            ground_truth in CLASSES
            and prediction in CLASSES
            and ground_truth != prediction
        ):
            wrong.append(item)

    print()
    print("=" * 70)
    print("WRONG WINDOWS")
    print("=" * 70)

    print(
        f"Total evaluation windows: {len(results)}"
    )

    print(
        f"Wrong windows found: {len(wrong)}"
    )

    if not wrong:
        print()
        print("No wrong windows found.")
        return

    # --------------------------------------------------------
    # Distribution of errors.
    # --------------------------------------------------------

    print()
    print("Error distribution:")

    error_counts = {}

    for item in wrong:

        key = (
            item["ground_truth"],
            item["prediction"],
        )

        error_counts[key] = (
            error_counts.get(key, 0) + 1
        )

    for (
        actual,
        predicted,
    ), count in sorted(error_counts.items()):

        print(
            f"  {actual} -> {predicted}: {count}"
        )

    # --------------------------------------------------------
    # Build SFT records.
    # --------------------------------------------------------

    sft_data = []

    for index, item in enumerate(
        wrong,
        start=1,
    ):

        video_path = item[
            "video_path"
        ]

        start = float(
            item["start"]
        )

        end = float(
            item["end"]
        )

        label = item[
            "ground_truth"
        ]

        clip_id = (
            Path(video_path).stem
            + f"_{start:.2f}_{end:.2f}"
        )

        assistant_json = {
            "classification": label,
            "evidence": make_evidence(
                label
            ),
            "incident_summary": make_summary(
                label
            ),
        }

        record = {
            "id": (
                f"targeted_error_{index:04d}"
            ),

            "video": video_path,

            "start": start,

            "end": end,

            "label": label,

            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                        },
                        {
                            "type": "text",
                            "text": PROMPT,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                assistant_json,
                                ensure_ascii=False,
                            ),
                        }
                    ],
                },
            ],

            "metadata": {
                "clip_id": clip_id,
                "source_video": video_path,
                "original_start": start,
                "original_end": end,
                "actual_label": label,
                "human_label": label,
                "previous_prediction": item.get(
                    "prediction"
                ),
                "previous_raw_output": item.get(
                    "raw_output"
                ),
                "targeted_error": True,
            },
        }

        sft_data.append(record)

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            sft_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TARGETED SFT DATASET CREATED")
    print("=" * 70)

    print()
    print(
        f"Records: {len(sft_data)}"
    )

    print()
    print("Training-label distribution:")

    label_counts = {
        label: 0
        for label in CLASSES
    }

    for item in sft_data:
        label_counts[
            item["label"]
        ] += 1

    for label in CLASSES:
        print(
            f"  {label:<15}"
            f"{label_counts[label]}"
        )

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These are HUMAN-GROUND-TRUTH corrections "
        "for model mistakes."
    )

    print(
        "Do NOT overwrite sft_train.json yet."
    )

    print(
        "Do NOT train yet."
    )

    print(
        "Review the generated dataset first."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()