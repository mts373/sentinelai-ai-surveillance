import json
from pathlib import Path
from collections import defaultdict, Counter


ROOT = Path(r"C:\SentinelAI_Qwen")

RESULTS_FILE = (
    ROOT
    / "dataset"
    / "test_evaluation"
    / "fresh_holdout"
    / "fresh_holdout_comparison.json"
)

OUTPUT_FILE = (
    ROOT
    / "dataset"
    / "test_evaluation"
    / "fresh_holdout"
    / "video_level_results.json"
)

CLASSES = [
    "Normal",
    "Fire",
    "Fight",
    "Road Accident",
]


def majority_vote(predictions):
    """
    Simple majority vote.

    If there is a tie, choose the prediction from
    the latest window among tied classes.
    """

    valid = [
        p for p in predictions
        if p in CLASSES
    ]

    if not valid:
        return None

    counts = Counter(valid)

    max_count = max(
        counts.values()
    )

    winners = [
        label
        for label, count in counts.items()
        if count == max_count
    ]

    if len(winners) == 1:
        return winners[0]

    # Tie-break:
    # use the latest prediction among tied classes.
    for prediction in reversed(valid):
        if prediction in winners:
            return prediction

    return winners[0]


def evaluate_video_level(
    model_name,
    results,
):

    groups = defaultdict(list)

    for item in results:

        prediction = item.get(
            "prediction"
        )

        truth = item.get(
            "ground_truth"
        )

        video = item.get(
            "video_path"
        )

        if (
            prediction not in CLASSES
            or truth not in CLASSES
            or not video
        ):
            continue

        groups[
            video
        ].append(item)

    video_results = []

    for video_path, windows in groups.items():

        predictions = [
            item["prediction"]
            for item in windows
        ]

        ground_truths = [
            item["ground_truth"]
            for item in windows
        ]

        # Human labels should be consistent within
        # a video for this evaluation setup.
        truth_counts = Counter(
            ground_truths
        )

        ground_truth = (
            truth_counts.most_common(1)[0][0]
        )

        video_prediction = majority_vote(
            predictions
        )

        correct = (
            video_prediction
            == ground_truth
        )

        video_results.append(
            {
                "video_path": video_path,
                "ground_truth": ground_truth,
                "window_count": len(windows),
                "window_predictions": predictions,
                "prediction_counts": dict(
                    Counter(predictions)
                ),
                "video_prediction":
                    video_prediction,
                "correct": correct,
            }
        )

    evaluated = len(
        video_results
    )

    correct = sum(
        item["correct"]
        for item in video_results
    )

    accuracy = (
        correct / evaluated
        if evaluated
        else 0.0
    )

    confusion = {
        actual: {
            predicted: 0
            for predicted in CLASSES
        }
        for actual in CLASSES
    }

    for item in video_results:

        actual = item[
            "ground_truth"
        ]

        predicted = item[
            "video_prediction"
        ]

        if predicted in CLASSES:

            confusion[
                actual
            ][
                predicted
            ] += 1

    per_class = {}

    for label in CLASSES:

        class_items = [
            item
            for item in video_results
            if item["ground_truth"]
            == label
        ]

        class_correct = sum(
            item["correct"]
            for item in class_items
        )

        class_accuracy = (
            class_correct
            / len(class_items)
            if class_items
            else 0.0
        )

        per_class[
            label
        ] = {
            "videos": len(
                class_items
            ),
            "correct": class_correct,
            "accuracy":
                class_accuracy,
            "accuracy_percent":
                class_accuracy * 100,
        }

    return {
        "model": model_name,
        "evaluated_videos": evaluated,
        "correct_videos": correct,
        "accuracy": accuracy,
        "accuracy_percent":
            accuracy * 100,
        "confusion_matrix":
            confusion,
        "per_class":
            per_class,
        "videos":
            video_results,
    }


def main():

    print()
    print("=" * 70)
    print(
        "SENTINELAI - VIDEO LEVEL "
        "FRESH HOLDOUT EVALUATION"
    )
    print("=" * 70)

    if not RESULTS_FILE.exists():

        raise FileNotFoundError(
            f"Results file not found:\n"
            f"{RESULTS_FILE}"
        )

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    baseline_windows = (
        data["baseline"]["results"]
    )

    corrective_windows = (
        data["corrective_v3"]["results"]
    )

    print()
    print(
        f"Window results loaded:"
    )

    print(
        f"  Original LoRA: "
        f"{len(baseline_windows)}"
    )

    print(
        f"  Corrective v3: "
        f"{len(corrective_windows)}"
    )

    # --------------------------------------------------------
    # Evaluate both models.
    # --------------------------------------------------------

    baseline = evaluate_video_level(
        "Original LoRA",
        baseline_windows,
    )

    corrective = evaluate_video_level(
        "Corrective v3",
        corrective_windows,
    )

    # --------------------------------------------------------
    # Display.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "VIDEO-LEVEL RESULTS"
    )
    print("=" * 70)

    print()

    print(
        f"Original LoRA:"
    )

    print(
        f"  Videos: "
        f"{baseline['evaluated_videos']}"
    )

    print(
        f"  Correct: "
        f"{baseline['correct_videos']}"
    )

    print(
        f"  Accuracy: "
        f"{baseline['accuracy_percent']:.2f}%"
    )

    print()

    print(
        f"Corrective v3:"
    )

    print(
        f"  Videos: "
        f"{corrective['evaluated_videos']}"
    )

    print(
        f"  Correct: "
        f"{corrective['correct_videos']}"
    )

    print(
        f"  Accuracy: "
        f"{corrective['accuracy_percent']:.2f}%"
    )

    improvement = (
        corrective["accuracy_percent"]
        - baseline["accuracy_percent"]
    )

    print()

    print(
        f"Improvement: "
        f"{improvement:+.2f} percentage points"
    )

    # --------------------------------------------------------
    # Per class.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PER-CLASS VIDEO PERFORMANCE"
    )
    print("=" * 70)

    print()

    print(
        f"{'Class':<18}"
        f"{'Original':>12}"
        f"{'Corrective':>14}"
    )

    print("-" * 44)

    for label in CLASSES:

        b = baseline[
            "per_class"
        ][label]

        c = corrective[
            "per_class"
        ][label]

        print(
            f"{label:<18}"
            f"{b['accuracy_percent']:>10.2f}%"
            f"{c['accuracy_percent']:>12.2f}%"
        )

    # --------------------------------------------------------
    # Print individual videos where models differ.
    # --------------------------------------------------------

    baseline_map = {
        item["video_path"]:
            item
        for item in baseline["videos"]
    }

    corrective_map = {
        item["video_path"]:
            item
        for item in corrective["videos"]
    }

    print()
    print("=" * 70)
    print(
        "MODEL DISAGREEMENTS"
    )
    print("=" * 70)

    disagreements = 0

    for video in baseline_map:

        if video not in corrective_map:
            continue

        b = baseline_map[
            video
        ]

        c = corrective_map[
            video
        ]

        if (
            b["video_prediction"]
            != c["video_prediction"]
        ):

            disagreements += 1

            print()
            print(
                Path(video).name
            )

            print(
                f"Actual: "
                f"{b['ground_truth']}"
            )

            print(
                f"Original: "
                f"{b['video_prediction']}"
            )

            print(
                f"Corrective: "
                f"{c['video_prediction']}"
            )

    if disagreements == 0:

        print(
            "No video-level disagreements."
        )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    output = {
        "evaluation":
            "fresh_holdout_video_level",
        "source":
            str(RESULTS_FILE),
        "original_lora":
            baseline,
        "corrective_v3":
            corrective,
        "comparison": {
            "original_accuracy_percent":
                baseline[
                    "accuracy_percent"
                ],
            "corrective_accuracy_percent":
                corrective[
                    "accuracy_percent"
                ],
            "improvement_percentage_points":
                improvement,
        },
    }

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
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print(
        "VIDEO-LEVEL EVALUATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "DO NOT TRAIN YET."
    )


if __name__ == "__main__":
    main()