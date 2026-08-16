import json
import random
from pathlib import Path
from collections import defaultdict


# ============================================================
# SENTINELAI - VIDEO-DIVERSE ACTIVE LEARNING SAMPLER
# ============================================================
#
# PURPOSE
# -------
# Create a small, diverse batch of temporal clips for HUMAN
# verification before SFT.
#
# IMPORTANT
# ---------
# Qwen predictions are used only as a weak prioritization
# signal. They are NEVER treated as ground-truth labels.
#
# Main goals:
#
# 1. Avoid repeatedly selecting clips from the same video.
# 2. Cover many different source videos.
# 3. Cover different temporal positions.
# 4. Prioritize anomaly videos.
# 5. Keep Normal examples limited because we already have
#    enough verified Normal clips.
#
# OUTPUT:
#
# dataset\sft_annotations\
#     video_diverse_candidates.json
# ============================================================


PROJECT_ROOT = Path(
    r"C:\SentinelAI_Qwen"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
)

CLIP_MANIFEST = (
    DATASET_ROOT
    / "clips"
    / "train.jsonl"
)

PILOT_RESULTS = (
    DATASET_ROOT
    / "temporal_labels"
    / "pilot_temporal_labels.json"
)

ANNOTATION_FILE = (
    DATASET_ROOT
    / "sft_annotations"
    / "train_annotations.json"
)

OUTPUT_ROOT = (
    DATASET_ROOT
    / "sft_annotations"
)

OUTPUT_FILE = (
    OUTPUT_ROOT
    / "video_diverse_candidates.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

# Number of NEW clips we want to manually verify.
#
# We deliberately keep this manageable.

TARGETS = {

    "Normal": 10,

    "Fire": 30,

    "Fight": 30,

    "Road Accident": 40,

}


CLASSES = [
    "Normal",
    "Fire",
    "Fight",
    "Road Accident",
]


ANOMALY_CLASSES = [
    "Fire",
    "Fight",
    "Road Accident",
]


# Maximum number of selected clips from ONE video.
#
# This prevents one video from dominating the batch.

MAX_CLIPS_PER_VIDEO = 2


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    if not path.exists():

        return []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# UNIQUE CLIP KEY
# ============================================================

def clip_key(
    video_path,
    start,
    end,
):

    return (
        str(video_path),
        round(
            float(start),
            3,
        ),
        round(
            float(end),
            3,
        ),
    )


# ============================================================
# LOAD EXISTING ANNOTATIONS
# ============================================================

def get_annotated_keys():

    annotations = load_json(
        ANNOTATION_FILE
    )

    keys = set()

    for item in annotations:

        if (
            "video_path" not in item
            or "start" not in item
            or "end" not in item
        ):
            continue

        keys.add(
            clip_key(
                item[
                    "video_path"
                ],
                item[
                    "start"
                ],
                item[
                    "end"
                ],
            )
        )

    return keys


# ============================================================
# LOAD QWEN PILOT PREDICTIONS
# ============================================================

def load_qwen_predictions():

    if not PILOT_RESULTS.exists():

        print()
        print(
            "WARNING: Qwen pilot file "
            "not found."
        )

        print(
            PILOT_RESULTS
        )

        return {}

    data = load_json(
        PILOT_RESULTS
    )

    predictions = {}

    for item in data:

        if (
            "video_path" not in item
            or "start" not in item
            or "end" not in item
        ):
            continue

        key = clip_key(
            item[
                "video_path"
            ],
            item[
                "start"
            ],
            item[
                "end"
            ],
        )

        predictions[
            key
        ] = item

    return predictions


# ============================================================
# TEMPORAL POSITION
# ============================================================

def temporal_position(
    start,
    end,
    video_duration=None,
):

    # Simple temporal bucket.
    #
    # We don't need exact video duration because the manifest
    # already gives us many temporal locations.

    start = float(start)

    if start < 15:

        return "early"

    if start < 45:

        return "middle"

    return "late"


# ============================================================
# SCORE CANDIDATE
# ============================================================

def score_candidate(
    record,
    qwen_prediction,
):

    source_label = record[
        "label"
    ]

    score = 0

    prediction = None

    if qwen_prediction:

        prediction = (
            qwen_prediction.get(
                "classification"
            )
        )

    # --------------------------------------------------------
    # Qwen agrees with source class.
    #
    # Useful candidate.
    # --------------------------------------------------------

    if (
        prediction
        == source_label
    ):

        score += 100

    # --------------------------------------------------------
    # Qwen detects another anomaly.
    #
    # Potentially interesting hard example.
    # --------------------------------------------------------

    elif prediction in ANOMALY_CLASSES:

        score += 80

    # --------------------------------------------------------
    # Qwen says Normal on anomaly source video.
    #
    # Potential false negative.
    # Human verification is important.
    # --------------------------------------------------------

    elif (
        source_label in ANOMALY_CLASSES
        and prediction == "Normal"
    ):

        score += 70

    # --------------------------------------------------------
    # Normal predicted Normal.
    # Useful negative.
    # --------------------------------------------------------

    elif (
        source_label == "Normal"
        and prediction == "Normal"
    ):

        score += 40

    # --------------------------------------------------------
    # Unknown / no Qwen prediction.
    # --------------------------------------------------------

    else:

        score += 20

    return score


# ============================================================
# ADD DIVERSITY BONUS
# ============================================================

def diversity_score(
    record,
    selected_video_counts,
    selected_temporal_buckets,
):

    video_path = record[
        "video_path"
    ]

    bucket = temporal_position(
        record[
            "start"
        ],
        record[
            "end"
        ],
    )

    score = 0

    # --------------------------------------------------------
    # Strong bonus for videos not selected yet.
    # --------------------------------------------------------

    if (
        selected_video_counts[
            video_path
        ] == 0
    ):

        score += 50

    # --------------------------------------------------------
    # Bonus for temporal diversity.
    # --------------------------------------------------------

    temporal_key = (
        video_path,
        bucket,
    )

    if (
        temporal_key
        not in selected_temporal_buckets
    ):

        score += 20

    # --------------------------------------------------------
    # Penalize videos already represented.
    # --------------------------------------------------------

    score -= (
        selected_video_counts[
            video_path
        ]
        * 25
    )

    return score


# ============================================================
# PREPARE CANDIDATES
# ============================================================

def prepare_candidates(
    records,
    predictions,
    annotated_keys,
):

    candidates = defaultdict(
        list
    )

    for record in records:

        label = record.get(
            "label"
        )

        if label not in CLASSES:

            continue

        key = clip_key(
            record[
                "video_path"
            ],
            record[
                "start"
            ],
            record[
                "end"
            ],
        )

        # ----------------------------------------------------
        # Never select already annotated clips.
        # ----------------------------------------------------

        if key in annotated_keys:

            continue

        qwen_prediction = (
            predictions.get(
                key
            )
        )

        base_score = score_candidate(
            record,
            qwen_prediction
        )

        candidate = {

            "video_path":
                record[
                    "video_path"
                ],

            "source_label":
                label,

            "clip_id":
                record[
                    "clip_id"
                ],

            "start":
                record[
                    "start"
                ],

            "end":
                record[
                    "end"
                ],

            "duration":
                record[
                    "duration"
                ],

            "split":
                record[
                    "split"
                ],

            "qwen_prediction":
                (
                    qwen_prediction.get(
                        "classification"
                    )
                    if qwen_prediction
                    else None
                ),

            "qwen_evidence":
                (
                    qwen_prediction.get(
                        "evidence"
                    )
                    if qwen_prediction
                    else None
                ),

            "base_score":
                base_score,

        }

        candidates[
            label
        ].append(
            candidate
        )

    return candidates


# ============================================================
# VIDEO-DIVERSE SELECTION
# ============================================================

def select_diverse(
    candidates,
    target,
):

    selected = []

    selected_video_counts = defaultdict(
        int
    )

    selected_temporal_buckets = set()

    # --------------------------------------------------------
    # Shuffle first so equal-score candidates don't always
    # produce the same result.
    # --------------------------------------------------------

    random.shuffle(
        candidates
    )

    remaining = candidates.copy()

    while (
        remaining
        and len(selected)
        < target
    ):

        scored = []

        for candidate in remaining:

            video_path = candidate[
                "video_path"
            ]

            # -----------------------------------------------
            # Hard maximum per video
            # -----------------------------------------------

            if (
                selected_video_counts[
                    video_path
                ]
                >= MAX_CLIPS_PER_VIDEO
            ):

                continue

            diversity = diversity_score(
                candidate,
                selected_video_counts,
                selected_temporal_buckets,
            )

            total_score = (
                candidate[
                    "base_score"
                ]
                + diversity
                + random.randint(
                    0,
                    9,
                )
            )

            scored.append(
                (
                    total_score,
                    candidate,
                )
            )

        if not scored:

            break

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        best_score, best = scored[
            0
        ]

        best[
            "selection_score"
        ] = best_score

        selected.append(
            best
        )

        video_path = best[
            "video_path"
        ]

        selected_video_counts[
            video_path
        ] += 1

        bucket = temporal_position(
            best[
                "start"
            ],
            best[
                "end"
            ],
        )

        selected_temporal_buckets.add(
            (
                video_path,
                bucket,
            )
        )

        remaining.remove(
            best
        )

    return selected


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(
        RANDOM_SEED
    )

    print()
    print("=" * 70)

    print(
        "SENTINELAI - VIDEO-DIVERSE "
        "ACTIVE LEARNING"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    records = load_jsonl(
        CLIP_MANIFEST
    )

    predictions = (
        load_qwen_predictions()
    )

    annotated_keys = (
        get_annotated_keys()
    )

    print(
        f"Manifest clips: "
        f"{len(records)}"
    )

    print(
        f"Qwen pilot predictions: "
        f"{len(predictions)}"
    )

    print(
        f"Already annotated: "
        f"{len(annotated_keys)}"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    candidates = prepare_candidates(
        records,
        predictions,
        annotated_keys,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "AVAILABLE CANDIDATES"
    )

    print(
        "=" * 70
    )

    for label in CLASSES:

        print(
            f"  {label:<15}"
            f"{len(candidates[label])}"
        )

    # --------------------------------------------------------
    # Select
    # --------------------------------------------------------

    all_selected = []

    print()
    print(
        "=" * 70
    )

    print(
        "SELECTING VIDEO-DIVERSE BATCH"
    )

    print(
        "=" * 70
    )

    for label in CLASSES:

        selected = select_diverse(
            candidates[
                label
            ],
            TARGETS[
                label
            ],
        )

        all_selected.extend(
            selected
        )

        videos = set(
            item[
                "video_path"
            ]
            for item in selected
        )

        qwen_counts = defaultdict(
            int
        )

        for item in selected:

            prediction = (
                item[
                    "qwen_prediction"
                ]
                or "Unknown"
            )

            qwen_counts[
                prediction
            ] += 1

        print()
        print(
            f"{label}"
        )

        print(
            f"  Target: "
            f"{TARGETS[label]}"
        )

        print(
            f"  Selected: "
            f"{len(selected)}"
        )

        print(
            f"  Unique videos: "
            f"{len(videos)}"
        )

        print(
            f"  Max clips/video: "
            f"{MAX_CLIPS_PER_VIDEO}"
        )

        print(
            "  Qwen predictions:"
        )

        for prediction, count in sorted(
            qwen_counts.items()
        ):

            print(
                f"    "
                f"{prediction:<15}"
                f"{count}"
            )

    # --------------------------------------------------------
    # Final shuffle
    # --------------------------------------------------------

    random.shuffle(
        all_selected
    )

    # Add annotation batch index

    for index, item in enumerate(
        all_selected,
        start=1,
    ):

        item[
            "annotation_index"
        ] = index

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_selected,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "VIDEO-DIVERSE SAMPLING COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Total candidates: "
        f"{len(all_selected)}"
    )

    print()

    print(
        "Target distribution:"
    )

    for label in CLASSES:

        count = sum(
            1
            for item in all_selected
            if item[
                "source_label"
            ] == label
        )

        print(
            f"  {label:<15}"
            f"{count}"
        )

    # --------------------------------------------------------
    # Unique video count
    # --------------------------------------------------------

    unique_videos = set(
        item[
            "video_path"
        ]
        for item in all_selected
    )

    print()

    print(
        f"Unique source videos: "
        f"{len(unique_videos)}"
    )

    print()

    print(
        f"Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "These are CANDIDATES only."
    )

    print(
        "The source label is NOT the final label."
    )

    print(
        "Watch each selected clip and assign:"
    )

    print(
        "Normal / Fire / Fight / Road Accident"
    )

    print()

    print(
        "Do NOT start LoRA training yet."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()