import json
import random
from pathlib import Path


# ============================================================
# SENTINELAI - TEMPORAL CLIP MANIFEST
# ============================================================
#
# Creates temporal training samples WITHOUT copying videos.
#
# Each record contains:
#
#   original video
#   start time
#   end time
#   class
#
# IMPORTANT:
# Train/validation/test video separation was already performed
# by prepare_dataset.py.
#
# Therefore this script NEVER moves a video between splits.
# ============================================================


SEED = 42

CLIP_DURATION = 10.0

# Number of temporal clips to generate per video.
#
# We intentionally keep this controlled.
# We do NOT generate every possible overlapping window.
#
CLIPS_PER_VIDEO = 3


DATASET_ROOT = Path(
    r"C:\SentinelAI_Qwen\dataset"
)


OUTPUT_ROOT = (
    DATASET_ROOT
    / "clips"
)


# ============================================================
# VIDEO INFORMATION
# ============================================================

import cv2


def get_duration(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        return None

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frames = cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    cap.release()

    if fps <= 0 or frames <= 0:

        return None

    return frames / fps


# ============================================================
# CREATE TEMPORAL WINDOWS
# ============================================================

def create_windows(
    duration,
    number_of_clips,
    rng,
):

    # --------------------------------------------------------
    # Very short video
    # --------------------------------------------------------

    if duration <= CLIP_DURATION:

        return [
            (
                0.0,
                round(duration, 2)
            )
        ]

    max_start = (
        duration
        - CLIP_DURATION
    )

    # --------------------------------------------------------
    # Deterministic candidate positions:
    #
    # beginning
    # middle
    # end
    #
    # Then randomize if more samples are required.
    # --------------------------------------------------------

    candidates = [

        0.0,

        max_start / 2.0,

        max_start,
    ]

    # --------------------------------------------------------
    # Additional random positions
    # --------------------------------------------------------

    while (
        len(candidates)
        < number_of_clips
    ):

        value = rng.uniform(
            0.0,
            max_start
        )

        candidates.append(
            value
        )

    # --------------------------------------------------------
    # Remove near-duplicates
    # --------------------------------------------------------

    unique = []

    for start in candidates:

        if all(
            abs(start - existing)
            >= 1.0
            for existing in unique
        ):

            unique.append(
                start
            )

        if len(unique) >= number_of_clips:

            break

    windows = []

    for start in unique:

        end = min(
            start + CLIP_DURATION,
            duration
        )

        windows.append(
            (
                round(start, 2),
                round(end, 2)
            )
        )

    return windows


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(
    split
):

    path = (
        DATASET_ROOT
        / f"{split}.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing dataset split:\n"
            f"{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# CREATE MANIFEST
# ============================================================

def create_manifest(
    split,
    rng,
):

    records = load_split(
        split
    )

    manifest = []

    print()
    print(
        f"Creating {split} "
        f"temporal manifest..."
    )

    for video_index, record in enumerate(
        records,
        start=1,
    ):

        video_path = Path(
            record[
                "video_path"
            ]
        )

        label = record[
            "label"
        ]

        duration = get_duration(
            video_path
        )

        if duration is None:

            print(
                f"WARNING: Could not read:"
            )

            print(
                f"  {video_path}"
            )

            continue

        windows = create_windows(
            duration,
            CLIPS_PER_VIDEO,
            rng
        )

        for clip_index, (
            start,
            end
        ) in enumerate(
            windows,
            start=1,
        ):

            manifest.append({

                "video_path":
                    str(video_path),

                "label":
                    label,

                "split":
                    split,

                "clip_id":
                    (
                        f"{video_path.stem}"
                        f"_{clip_index:02d}"
                    ),

                "start":
                    start,

                "end":
                    end,

                "duration":
                    round(
                        end - start,
                        2
                    ),
            })

        if (
            video_index % 50
            == 0
        ):

            print(
                f"  Processed "
                f"{video_index}/"
                f"{len(records)} videos"
            )

    return manifest


# ============================================================
# SAVE MANIFEST
# ============================================================

def save_jsonl(
    path,
    records,
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

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    split,
    records,
):

    counts = {

        "Normal": 0,

        "Fire": 0,

        "Fight": 0,

        "Road Accident": 0,
    }

    for record in records:

        label = record[
            "label"
        ]

        if label in counts:

            counts[label] += 1

    print()
    print(
        f"{split.upper()} CLIPS"
    )

    for label, count in counts.items():

        print(
            f"  {label:<15} "
            f"{count}"
        )

    print(
        f"  {'TOTAL':<15} "
        f"{len(records)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SENTINELAI - TEMPORAL "
        "CLIP MANIFEST CREATION"
    )
    print("=" * 70)

    print(
        f"Dataset: "
        f"{DATASET_ROOT}"
    )

    print(
        f"Clip duration: "
        f"{CLIP_DURATION}s"
    )

    print(
        f"Clips per video: "
        f"{CLIPS_PER_VIDEO}"
    )

    print(
        f"Seed: {SEED}"
    )

    print()

    rng = random.Random(
        SEED
    )

    all_statistics = {}

    for split in [
        "train",
        "val",
        "test",
    ]:

        manifest = create_manifest(
            split,
            rng
        )

        output_path = (
            OUTPUT_ROOT
            / f"{split}.jsonl"
        )

        save_jsonl(
            output_path,
            manifest
        )

        print_statistics(
            split,
            manifest
        )

        print(
            f"Saved: "
            f"{output_path}"
        )

        all_statistics[
            split
        ] = len(manifest)

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "seed":
            SEED,

        "clip_duration":
            CLIP_DURATION,

        "clips_per_video":
            CLIPS_PER_VIDEO,

        "train_clips":
            all_statistics["train"],

        "validation_clips":
            all_statistics["val"],

        "test_clips":
            all_statistics["test"],
    }

    summary_path = (
        OUTPUT_ROOT
        / "clip_manifest_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    print()
    print("=" * 70)
    print(
        "TEMPORAL MANIFEST COMPLETE"
    )
    print("=" * 70)

    print(
        f"Train clips: "
        f"{all_statistics['train']}"
    )

    print(
        f"Validation clips: "
        f"{all_statistics['val']}"
    )

    print(
        f"Test clips: "
        f"{all_statistics['test']}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No videos were copied or modified."
    )

    print(
        "The manifest only stores "
        "video paths and timestamps."
    )

    print(
        f"\nSummary: {summary_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()