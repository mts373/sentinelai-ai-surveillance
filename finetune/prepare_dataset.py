import json
import random
from pathlib import Path


# ============================================================
# SENTINELAI - DATASET PREPARATION
# ============================================================
#
# Four classes ONLY:
#
#   Normal
#   Fire
#   Fight
#   Road Accident
#
# IMPORTANT:
# We split ORIGINAL VIDEOS first.
# Temporal clips will be generated later.
#
# This prevents data leakage between train/validation/test.
# ============================================================


SEED = 42

DATASET_ROOT = Path(
    r"C:\UCF_CRIME_ORIGINAL"
)

OUTPUT_ROOT = Path(
    r"C:\SentinelAI_Qwen\dataset"
)


# ============================================================
# SOURCE DIRECTORIES
# ============================================================

CLASS_DIRECTORIES = {

    "Fire": [
        DATASET_ROOT
        / "Anomaly-Videos-Part-1"
        / "Arson"
    ],

    "Fight": [
        DATASET_ROOT
        / "Anomaly-Videos-Part-2"
        / "Fighting"
    ],

    "Road Accident": [
        DATASET_ROOT
        / "Anomaly-Videos-Part-3"
        / "RoadAccidents"
    ],

    "Normal": [
        DATASET_ROOT
        / "Testing_Normal_Videos_Anomaly",

        DATASET_ROOT
        / "Training-Normal-Videos-Part-1",

        DATASET_ROOT
        / "Training-Normal-Videos-Part-2",
    ],
}


# ============================================================
# TARGET NUMBER OF ORIGINAL VIDEOS
#
# We deliberately don't use all 950 Normal videos.
# This prevents Normal from dominating the training set.
# ============================================================

TARGET_VIDEOS = {

    "Normal": 450,

    "Fire": 50,

    "Fight": 50,

    "Road Accident": 150,
}


# ============================================================
# SPLIT RATIOS
#
# These are approximate:
#
# Train = 80%
# Validation = 10%
# Test = 10%
#
# The split is performed on ORIGINAL VIDEOS.
# ============================================================

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================================
# VIDEO EXTENSIONS
# ============================================================

VIDEO_EXTENSIONS = {

    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# ============================================================
# FIND VIDEOS
# ============================================================

def find_videos(
    directories
):

    videos = []

    for directory in directories:

        if not directory.exists():

            print(
                f"WARNING: Directory not found:"
            )

            print(
                f"  {directory}"
            )

            continue

        for path in directory.rglob("*"):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                in VIDEO_EXTENSIONS
            ):

                videos.append(
                    path.resolve()
                )

    # Remove duplicates while preserving
    # deterministic ordering.

    videos = sorted(
        set(videos),
        key=lambda p: str(p).lower()
    )

    return videos


# ============================================================
# SELECT VIDEOS
# ============================================================

def select_videos(
    videos,
    target,
    rng,
):

    if target >= len(videos):

        return videos.copy()

    selected = rng.sample(
        videos,
        target
    )

    return sorted(
        selected,
        key=lambda p: str(p).lower()
    )


# ============================================================
# SPLIT VIDEOS
# ============================================================

def split_videos(
    videos,
    rng,
):

    shuffled = videos.copy()

    rng.shuffle(
        shuffled
    )

    total = len(
        shuffled
    )

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    train = shuffled[
        :train_count
    ]

    val = shuffled[
        train_count:
        train_count + val_count
    ]

    test = shuffled[
        train_count + val_count:
    ]

    return (
        train,
        val,
        test
    )


# ============================================================
# CREATE RECORD
# ============================================================

def make_record(
    path,
    label,
    split,
):

    return {

        "video_path":
            str(path),

        "label":
            label,

        "split":
            split,

    }


# ============================================================
# WRITE JSON
# ============================================================

def write_json(
    path,
    data,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SENTINELAI - DATASET PREPARATION"
    )
    print("=" * 70)

    print(
        f"Dataset root: "
        f"{DATASET_ROOT}"
    )

    print(
        f"Output root: "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Random seed: "
        f"{SEED}"
    )

    print()

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"Dataset root not found:\n"
            f"{DATASET_ROOT}"
        )

    rng = random.Random(
        SEED
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test",
    ]:

        (
            OUTPUT_ROOT
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    all_records = {

        "train": [],

        "val": [],

        "test": [],
    }

    summary = {

        "seed":
            SEED,

        "classes": [
            "Normal",
            "Fire",
            "Fight",
            "Road Accident",
        ],

        "total_original_videos":
            0,

        "classes_summary": {},

        "train":
            {},

        "val":
            {},

        "test":
            {},
    }

    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for label in [
        "Normal",
        "Fire",
        "Fight",
        "Road Accident",
    ]:

        print("-" * 70)

        print(
            f"CLASS: {label}"
        )

        directories = (
            CLASS_DIRECTORIES[
                label
            ]
        )

        videos = find_videos(
            directories
        )

        print(
            f"Found: "
            f"{len(videos)} videos"
        )

        target = TARGET_VIDEOS[
            label
        ]

        selected = select_videos(
            videos,
            target,
            rng
        )

        print(
            f"Selected: "
            f"{len(selected)} videos"
        )

        if len(selected) < target:

            print(
                f"WARNING: Requested "
                f"{target}, but only "
                f"{len(selected)} available."
            )

        train, val, test = (
            split_videos(
                selected,
                rng
            )
        )

        print(
            f"Train: "
            f"{len(train)}"
        )

        print(
            f"Validation: "
            f"{len(val)}"
        )

        print(
            f"Test: "
            f"{len(test)}"
        )

        # ----------------------------------------------------
        # Records
        # ----------------------------------------------------

        for path in train:

            all_records[
                "train"
            ].append(
                make_record(
                    path,
                    label,
                    "train"
                )
            )

        for path in val:

            all_records[
                "val"
            ].append(
                make_record(
                    path,
                    label,
                    "val"
                )
            )

        for path in test:

            all_records[
                "test"
            ].append(
                make_record(
                    path,
                    label,
                    "test"
                )
            )

        summary[
            "classes_summary"
        ][label] = {

            "available":
                len(videos),

            "selected":
                len(selected),

            "train":
                len(train),

            "validation":
                len(val),

            "test":
                len(test),
        }

        summary[
            "train"
        ][label] = len(train)

        summary[
            "val"
        ][label] = len(val)

        summary[
            "test"
        ][label] = len(test)

    # ========================================================
    # SHUFFLE RECORDS
    # ========================================================

    for split in all_records:

        rng.shuffle(
            all_records[
                split
            ]
        )

    # ========================================================
    # TOTALS
    # ========================================================

    summary[
        "total_original_videos"
    ] = sum(
        len(all_records[split])
        for split in all_records
    )

    summary[
        "train_total"
    ] = len(
        all_records["train"]
    )

    summary[
        "val_total"
    ] = len(
        all_records["val"]
    )

    summary[
        "test_total"
    ] = len(
        all_records["test"]
    )

    # ========================================================
    # WRITE MANIFESTS
    # ========================================================

    write_json(
        OUTPUT_ROOT
        / "train.json",
        all_records["train"]
    )

    write_json(
        OUTPUT_ROOT
        / "val.json",
        all_records["val"]
    )

    write_json(
        OUTPUT_ROOT
        / "test.json",
        all_records["test"]
    )

    write_json(
        OUTPUT_ROOT
        / "dataset_summary.json",
        summary
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "DATASET SPLIT COMPLETE"
    )
    print("=" * 70)

    print()

    print(
        f"Total videos: "
        f"{summary['total_original_videos']}"
    )

    print()

    print(
        "TRAIN"
    )

    for label in summary[
        "classes"
    ]:

        print(
            f"  {label:<15} "
            f"{summary['train'][label]}"
        )

    print()

    print(
        "VALIDATION"
    )

    for label in summary[
        "classes"
    ]:

        print(
            f"  {label:<15} "
            f"{summary['val'][label]}"
        )

    print()

    print(
        "TEST"
    )

    for label in summary[
        "classes"
    ]:

        print(
            f"  {label:<15} "
            f"{summary['test'][label]}"
        )

    print()

    print(
        f"Train total: "
        f"{summary['train_total']}"
    )

    print(
        f"Validation total: "
        f"{summary['val_total']}"
    )

    print(
        f"Test total: "
        f"{summary['test_total']}"
    )

    print()

    print(
        "Files created:"
    )

    print(
        f"  {OUTPUT_ROOT / 'train.json'}"
    )

    print(
        f"  {OUTPUT_ROOT / 'val.json'}"
    )

    print(
        f"  {OUTPUT_ROOT / 'test.json'}"
    )

    print(
        f"  {OUTPUT_ROOT / 'dataset_summary.json'}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Original videos were split BEFORE "
        "temporal clips are generated."
    )

    print(
        "This prevents train/validation/test "
        "video leakage."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()