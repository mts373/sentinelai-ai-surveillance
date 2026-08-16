import json
import cv2
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# SENTINELAI - SFT DATASET AUDIT
# ============================================================
#
# Checks:
#
# 1. JSON structure
# 2. Video files exist
# 3. Video files can be opened
# 4. Clip duration
# 5. Resolution / FPS / frame count
# 6. Duplicate clips
# 7. Source-video distribution
# 8. Class distribution
# 9. Suspicious temporal metadata
# 10. Train/validation/test video leakage
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\SentinelAI_Qwen"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
)

SFT_FILE = (
    DATASET_ROOT
    / "sft"
    / "sft_train.json"
)

CLIPS_ROOT = (
    DATASET_ROOT
    / "sft"
    / "clips"
)

TRAIN_MANIFEST = (
    DATASET_ROOT
    / "clips"
    / "train.jsonl"
)

VAL_MANIFEST = (
    DATASET_ROOT
    / "clips"
    / "val.jsonl"
)

TEST_MANIFEST = (
    DATASET_ROOT
    / "clips"
    / "test.jsonl"
)


# ============================================================
# EXPECTED CLASSES
# ============================================================

CLASSES = [

    "Normal",

    "Fire",

    "Fight",

    "Road Accident",

]


# ============================================================
# EXPECTED CONFIGURATION
# ============================================================

EXPECTED_TOTAL = 108

EXPECTED_COUNTS = {

    "Normal": 60,

    "Fire": 10,

    "Fight": 15,

    "Road Accident": 23,

}


MIN_DURATION = 8.0

MAX_DURATION = 12.0

SUSPICIOUS_SOURCE_TIME = 3600.0


# ============================================================
# LOAD JSON
# ============================================================

def load_json(
    path
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# LOAD JSONL MANIFEST
# ============================================================

def load_jsonl(
    path
):

    records = []

    if not path.exists():

        print(
            f"WARNING: Manifest not found:"
        )

        print(
            f"  {path}"
        )

        return records

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:

                continue

            try:

                records.append(
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError as error:

                print(
                    f"WARNING: Invalid JSONL "
                    f"at {path}:{line_number}"
                )

                print(
                    error
                )

    return records


# ============================================================
# NORMALIZE PATH
# ============================================================

def normalize_path(
    path
):

    try:

        return str(
            Path(
                path
            ).resolve()
        ).lower()

    except Exception:

        return str(
            path
        ).lower()


# ============================================================
# BUILD SPLIT VIDEO SETS
# ============================================================

def build_split_sets():

    train_records = load_jsonl(
        TRAIN_MANIFEST
    )

    val_records = load_jsonl(
        VAL_MANIFEST
    )

    test_records = load_jsonl(
        TEST_MANIFEST
    )

    train_videos = {

        normalize_path(
            item[
                "video_path"
            ]
        )

        for item in train_records

        if "video_path" in item

    }

    val_videos = {

        normalize_path(
            item[
                "video_path"
            ]
        )

        for item in val_records

        if "video_path" in item

    }

    test_videos = {

        normalize_path(
            item[
                "video_path"
            ]
        )

        for item in test_records

        if "video_path" in item

    }

    return (
        train_videos,
        val_videos,
        test_videos,
    )


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "SENTINELAI - SFT DATASET AUDIT"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "SFT dataset:"
    )

    print(
        SFT_FILE
    )

    print()

    # --------------------------------------------------------
    # Check SFT file
    # --------------------------------------------------------

    if not SFT_FILE.exists():

        raise FileNotFoundError(
            f"SFT dataset not found:\n"
            f"{SFT_FILE}"
        )

    dataset = load_json(
        SFT_FILE
    )

    if not isinstance(
        dataset,
        list,
    ):

        raise ValueError(
            "sft_train.json must "
            "contain a JSON list."
        )

    print(
        f"Total SFT records: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    class_counts = Counter()

    source_video_counts = Counter()

    clip_paths = set()

    duplicate_clips = []

    missing_clips = []

    unreadable_clips = []

    invalid_duration = []

    suspicious_times = []

    invalid_records = []

    resolution_counts = Counter()

    fps_counts = Counter()

    duration_values = []

    # --------------------------------------------------------
    # Record audit
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "AUDITING SFT RECORDS"
    )

    print(
        "=" * 70
    )

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        required_fields = [

            "id",

            "video",

            "label",

            "messages",

            "metadata",

        ]

        missing_fields = [

            field

            for field in required_fields

            if field not in item

        ]

        if missing_fields:

            invalid_records.append(
                (
                    index,
                    f"Missing fields: "
                    f"{missing_fields}"
                )
            )

            continue

        label = item[
            "label"
        ]

        if label not in CLASSES:

            invalid_records.append(
                (
                    index,
                    f"Invalid class: "
                    f"{label}"
                )
            )

        else:

            class_counts[
                label
            ] += 1

        # ----------------------------------------------------
        # Video path
        # ----------------------------------------------------

        video_path = Path(
            item[
                "video"
            ]
        )

        normalized_clip = (
            normalize_path(
                video_path
            )
        )

        if normalized_clip in clip_paths:

            duplicate_clips.append(
                (
                    index,
                    str(
                        video_path
                    )
                )
            )

        else:

            clip_paths.add(
                normalized_clip
            )

        # ----------------------------------------------------
        # Check existence
        # ----------------------------------------------------

        if not video_path.exists():

            missing_clips.append(
                (
                    index,
                    str(
                        video_path
                    )
                )
            )

            continue

        # ----------------------------------------------------
        # Open video
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            unreadable_clips.append(
                (
                    index,
                    str(
                        video_path
                    )
                )
            )

            cap.release()

            continue

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        duration = (

            frames / fps

            if fps > 0

            else 0.0

        )

        cap.release()

        duration_values.append(
            duration
        )

        resolution_counts[
            f"{width}x{height}"
        ] += 1

        fps_counts[
            round(
                fps,
                2
            )
        ] += 1

        # ----------------------------------------------------
        # Duration check
        # ----------------------------------------------------

        if (

            duration < MIN_DURATION

            or duration > MAX_DURATION

        ):

            invalid_duration.append(
                (
                    index,
                    str(
                        video_path
                    ),
                    duration
                )
            )

        # ----------------------------------------------------
        # Original temporal metadata
        # ----------------------------------------------------

        metadata = item[
            "metadata"
        ]

        original_start = metadata.get(
            "original_start"
        )

        original_end = metadata.get(
            "original_end"
        )

        if (

            original_start is not None

            and float(
                original_start
            ) >= SUSPICIOUS_SOURCE_TIME

        ):

            suspicious_times.append(
                (
                    index,
                    str(
                        video_path
                    ),
                    original_start,
                    original_end,
                )
            )

        # ----------------------------------------------------
        # Source video
        # ----------------------------------------------------

        source_video = metadata.get(
            "source_video"
        )

        if source_video:

            source_video_counts[
                normalize_path(
                    source_video
                )
            ] += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (

            index % 20 == 0

            or index == len(dataset)

        ):

            print(
                f"Audited "
                f"{index}/"
                f"{len(dataset)}"
            )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "CLASS DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    class_distribution_ok = True

    for label in CLASSES:

        actual = class_counts[
            label
        ]

        expected = EXPECTED_COUNTS[
            label
        ]

        status = (
            "OK"
            if actual == expected
            else "MISMATCH"
        )

        if actual != expected:

            class_distribution_ok = False

        print(
            f"{label:<15}"
            f"Expected: "
            f"{expected:<4}"
            f"Actual: "
            f"{actual:<4}"
            f"{status}"
        )

    # ========================================================
    # FILE CHECKS
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "FILE INTEGRITY"
    )

    print(
        "=" * 70
    )

    print(
        f"Missing clips: "
        f"{len(missing_clips)}"
    )

    print(
        f"Unreadable clips: "
        f"{len(unreadable_clips)}"
    )

    print(
        f"Duplicate clip paths: "
        f"{len(duplicate_clips)}"
    )

    print(
        f"Invalid durations: "
        f"{len(invalid_duration)}"
    )

    print(
        f"Invalid JSON records: "
        f"{len(invalid_records)}"
    )

    # ========================================================
    # RESOLUTION
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "RESOLUTION DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    for resolution, count in (
        resolution_counts.most_common()
    ):

        print(
            f"  {resolution:<12}"
            f"{count}"
        )

    # ========================================================
    # FPS
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "FPS DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    for fps, count in (
        fps_counts.most_common()
    ):

        print(
            f"  {fps:<10}"
            f"{count}"
        )

    # ========================================================
    # DURATION
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "CLIP DURATION"
    )

    print(
        "=" * 70
    )

    if duration_values:

        minimum = min(
            duration_values
        )

        maximum = max(
            duration_values
        )

        average = (
            sum(
                duration_values
            )
            /
            len(
                duration_values
            )
        )

        print(
            f"Minimum: "
            f"{minimum:.3f}s"
        )

        print(
            f"Maximum: "
            f"{maximum:.3f}s"
        )

        print(
            f"Average: "
            f"{average:.3f}s"
        )

    # ========================================================
    # SUSPICIOUS SOURCE TIMES
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "SUSPICIOUS ORIGINAL TIMESTAMPS"
    )

    print(
        "=" * 70
    )

    if suspicious_times:

        print(
            f"Found: "
            f"{len(suspicious_times)}"
        )

        print()

        for (
            index,
            video,
            start,
            end,
        ) in suspicious_times:

            print(
                f"[{index}] "
                f"{Path(video).name}"
            )

            print(
                f"    "
                f"{start:.2f}s → "
                f"{end:.2f}s"
            )

    else:

        print(
            "None found."
        )

    # ========================================================
    # SOURCE VIDEO DIVERSITY
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "SOURCE VIDEO DIVERSITY"
    )

    print(
        "=" * 70
    )

    print(
        f"Unique source videos: "
        f"{len(source_video_counts)}"
    )

    repeated_sources = {

        path: count

        for path, count
        in source_video_counts.items()

        if count > 1

    }

    print(
        f"Source videos contributing "
        f"multiple clips: "
        f"{len(repeated_sources)}"
    )

    if repeated_sources:

        print()

        for path, count in sorted(
            repeated_sources.items(),
            key=lambda x: x[1],
            reverse=True,
        ):

            print(
                f"  {Path(path).name}: "
                f"{count} clips"
            )

    # ========================================================
    # TRAIN / VAL / TEST LEAKAGE
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "TRAIN / VALIDATION / TEST LEAKAGE"
    )

    print(
        "=" * 70
    )

    (
        train_videos,
        val_videos,
        test_videos,
    ) = build_split_sets()

    print(
        f"Train source videos: "
        f"{len(train_videos)}"
    )

    print(
        f"Validation source videos: "
        f"{len(val_videos)}"
    )

    print(
        f"Test source videos: "
        f"{len(test_videos)}"
    )

    sft_source_videos = {

        normalize_path(
            item[
                "metadata"
            ].get(
                "source_video"
            )
        )

        for item in dataset

        if item[
            "metadata"
        ].get(
            "source_video"
        )

    }

    train_overlap = (
        sft_source_videos
        & train_videos
    )

    val_overlap = (
        sft_source_videos
        & val_videos
    )

    test_overlap = (
        sft_source_videos
        & test_videos
    )

    print()

    print(
        f"SFT ∩ Train: "
        f"{len(train_overlap)}"
    )

    print(
        f"SFT ∩ Validation: "
        f"{len(val_overlap)}"
    )

    print(
        f"SFT ∩ Test: "
        f"{len(test_overlap)}"
    )

    if val_overlap:

        print()

        print(
            "WARNING: SFT / VALIDATION "
            "VIDEO LEAKAGE DETECTED"
        )

        for path in sorted(
            val_overlap
        ):

            print(
                f"  {Path(path).name}"
            )

    if test_overlap:

        print()

        print(
            "WARNING: SFT / TEST "
            "VIDEO LEAKAGE DETECTED"
        )

        for path in sorted(
            test_overlap
        ):

            print(
                f"  {Path(path).name}"
            )

    # ========================================================
    # SAMPLE RECORD
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "SAMPLE RECORD"
    )

    print(
        "=" * 70
    )

    if dataset:

        sample = dataset[0]

        print()

        print(
            f"ID: "
            f"{sample.get('id')}"
        )

        print(
            f"Label: "
            f"{sample.get('label')}"
        )

        print(
            f"Training clip:"
        )

        print(
            sample.get(
                "video"
            )
        )

        metadata = sample.get(
            "metadata",
            {}
        )

        print(
            f"Original video:"
        )

        print(
            metadata.get(
                "source_video"
            )
        )

        print(
            f"Original temporal window: "
            f"{metadata.get('original_start')} "
            f"→ "
            f"{metadata.get('original_end')}"
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    problems = []

    if len(dataset) != EXPECTED_TOTAL:

        problems.append(
            "Total SFT examples "
            "does not equal 108."
        )

    if not class_distribution_ok:

        problems.append(
            "Class distribution "
            "does not match target."
        )

    if missing_clips:

        problems.append(
            "Missing clips found."
        )

    if unreadable_clips:

        problems.append(
            "Unreadable clips found."
        )

    if duplicate_clips:

        problems.append(
            "Duplicate clips found."
        )

    if invalid_duration:

        problems.append(
            "Invalid clip durations found."
        )

    if invalid_records:

        problems.append(
            "Invalid JSON records found."
        )

    if val_overlap:

        problems.append(
            "SFT/validation video leakage."
        )

    if test_overlap:

        problems.append(
            "SFT/test video leakage."
        )

    # Suspicious timestamps are warnings,
    # not automatically fatal.

    if suspicious_times:

        problems.append(
            "Suspicious original timestamps "
            "require manual inspection."
        )

    print()

    print(
        "=" * 70
    )

    print(
        "FINAL AUDIT RESULT"
    )

    print(
        "=" * 70
    )

    if problems:

        print()

        print(
            "AUDIT STATUS: REVIEW REQUIRED"
        )

        print()

        for number, problem in enumerate(
            problems,
            start=1,
        ):

            print(
                f"{number}. {problem}"
            )

        print()

        print(
            "DO NOT START LoRA TRAINING YET."
        )

    else:

        print()

        print(
            "AUDIT STATUS: PASSED"
        )

        print()

        print(
            "The SFT dataset passed "
            "all automated checks."
        )

        print()

        print(
            "Next step:"
        )

        print(
            "Inspect the SFT samples "
            "before starting LoRA."
        )

    print()

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()