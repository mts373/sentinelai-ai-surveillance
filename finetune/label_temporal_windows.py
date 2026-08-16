import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path


# ============================================================
# ADD PROJECT SRC DIRECTORY TO PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(
    r"C:\SentinelAI_Qwen"
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

sys.path.insert(
    0,
    str(SRC_DIR)
)


import cv2
import torch

from qwen_engine import (
    load_model,
    qwen_generate_video,
    parse_json_output,
)


# ============================================================
# SENTINELAI - TEMPORAL LABELING PILOT
# ============================================================
#
# Purpose:
#
# Find which temporal windows of anomaly videos actually
# contain the anomaly.
#
# Classes:
#
#   Fire
#   Fight
#   Road Accident
#
# IMPORTANT:
#
# This is ONLY a pilot.
#
# We process a small number of videos first.
# We do NOT train anything yet.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\SentinelAI_Qwen"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
)

OUTPUT_ROOT = (
    DATASET_ROOT
    / "temporal_labels"
)


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SECONDS = 10.0

STEP_SECONDS = 5.0

DEFAULT_VIDEOS_PER_CLASS = 3

CLASSES = [
    "Fire",
    "Fight",
    "Road Accident",
]


# ============================================================
# GPU CLEANUP
# ============================================================

def cleanup_gpu():

    gc.collect()

    if not torch.cuda.is_available():

        return

    torch.cuda.empty_cache()

    try:

        torch.cuda.ipc_collect()

    except Exception:

        pass


# ============================================================
# NORMALIZE CLASSIFICATION
# ============================================================

def normalize_classification(
    value
):

    if value is None:

        return "Normal"

    text = str(
        value
    ).strip().lower()

    if (
        "road" in text
        and "accident" in text
    ):

        return "Road Accident"

    if "accident" in text:

        return "Road Accident"

    if "fight" in text:

        return "Fight"

    if "fire" in text:

        return "Fire"

    if "arson" in text:

        return "Fire"

    return "Normal"


# ============================================================
# CREATE TEMPORAL WINDOWS
# ============================================================

def create_windows(
    duration
):

    windows = []

    start = 0.0

    while start < duration:

        end = min(
            start + WINDOW_SECONDS,
            duration
        )

        windows.append({

            "start":
                round(
                    start,
                    2
                ),

            "end":
                round(
                    end,
                    2
                ),

        })

        if end >= duration:

            break

        start += STEP_SECONDS

    return windows


# ============================================================
# GET VIDEO DURATION
# ============================================================

def get_duration(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frames = cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    cap.release()

    if fps <= 0:

        fps = 30.0

    if frames <= 0:

        raise RuntimeError(
            f"Could not determine "
            f"frame count:\n"
            f"{video_path}"
        )

    return frames / fps


# ============================================================
# LOAD TRAINING VIDEO SPLIT
# ============================================================

def load_train_split():

    path = (
        DATASET_ROOT
        / "train.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Training manifest not found:\n"
            f"{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# SELECT PILOT VIDEOS
# ============================================================

def select_videos(
    records,
    videos_per_class
):

    selected = []

    for label in CLASSES:

        candidates = [

            record

            for record in records

            if record.get(
                "label"
            ) == label

        ]

        candidates = sorted(
            candidates,
            key=lambda item:
                item["video_path"]
        )

        selected.extend(
            candidates[
                :videos_per_class
            ]
        )

    return selected


# ============================================================
# EXTRACT TEMPORARY VIDEO WINDOW
# ============================================================

def extract_window(
    video_path,
    start,
    end,
):

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    )

    temp_path = Path(
        temp_file.name
    )

    temp_file.close()

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open:\n"
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0

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

    start_frame = int(
        start * fps
    )

    end_frame = int(
        end * fps
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height
        ),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create "
            "temporary video."
        )

    frame_number = (
        start_frame
    )

    try:

        while (
            frame_number
            < end_frame
        ):

            ret, frame = (
                cap.read()
            )

            if not ret:

                break

            writer.write(
                frame
            )

            frame_number += 1

    finally:

        cap.release()

        writer.release()

    return temp_path


# ============================================================
# ANALYZE ONE TEMPORAL WINDOW
# ============================================================

def analyze_window(
    model,
    processor,
    video_path,
    start,
    end,
):

    temp_path = None

    try:

        temp_path = extract_window(
            video_path,
            start,
            end,
        )

        raw = qwen_generate_video(
            model=model,
            processor=processor,
            video_path=temp_path,
        )

        parsed = parse_json_output(
            raw
        )

        classification = (
            normalize_classification(
                parsed.get(
                    "classification"
                )
            )
        )

        evidence = str(
            parsed.get(
                "evidence",
                ""
            )
        ).strip()

        incident_summary = str(
            parsed.get(
                "incident_summary",
                ""
            )
        ).strip()

        return {

            "classification":
                classification,

            "evidence":
                evidence,

            "incident_summary":
                incident_summary,

        }

    finally:

        if temp_path is not None:

            try:

                temp_path.unlink(
                    missing_ok=True
                )

            except Exception:

                pass

        cleanup_gpu()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "SentinelAI temporal "
            "labeling pilot"
        )
    )

    parser.add_argument(
        "--videos-per-class",
        type=int,
        default=DEFAULT_VIDEOS_PER_CLASS,
        help=(
            "Number of training videos "
            "to inspect per anomaly class"
        ),
    )

    args = parser.parse_args()

    videos_per_class = (
        args.videos_per_class
    )

    if videos_per_class < 1:

        raise ValueError(
            "videos-per-class "
            "must be at least 1."
        )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 70)
    print(
        "SENTINELAI - TEMPORAL "
        "LABELING PILOT"
    )
    print("=" * 70)

    print(
        f"Videos per class: "
        f"{videos_per_class}"
    )

    print(
        f"Window: "
        f"{WINDOW_SECONDS}s"
    )

    print(
        f"Step: "
        f"{STEP_SECONDS}s"
    )

    print()

    # ========================================================
    # LOAD DATASET
    # ========================================================

    records = load_train_split()

    selected = select_videos(
        records,
        videos_per_class,
    )

    print(
        f"Selected videos: "
        f"{len(selected)}"
    )

    for label in CLASSES:

        count = sum(
            1
            for record in selected
            if record["label"] == label
        )

        print(
            f"  {label:<15}"
            f"{count}"
        )

    print()

    # ========================================================
    # LOAD QWEN ONCE
    # ========================================================

    print(
        "Loading Qwen2.5-VL..."
    )

    model, processor = (
        load_model()
    )

    print(
        "Qwen loaded."
    )

    results = []

    try:

        # ====================================================
        # PROCESS VIDEOS
        # ====================================================

        for video_index, record in enumerate(
            selected,
            start=1,
        ):

            video_path = Path(
                record[
                    "video_path"
                ]
            )

            source_label = record[
                "label"
            ]

            if not video_path.exists():

                print(
                    f"WARNING: video "
                    f"does not exist:"
                )

                print(
                    f"  {video_path}"
                )

                continue

            duration = get_duration(
                video_path
            )

            windows = create_windows(
                duration
            )

            print()
            print("=" * 70)

            print(
                f"VIDEO "
                f"[{video_index}/"
                f"{len(selected)}]"
            )

            print(
                f"Expected class: "
                f"{source_label}"
            )

            print(
                f"Video: "
                f"{video_path.name}"
            )

            print(
                f"Duration: "
                f"{duration:.2f}s"
            )

            print(
                f"Windows: "
                f"{len(windows)}"
            )

            print("=" * 70)

            for window_index, window in enumerate(
                windows,
                start=1,
            ):

                start = window[
                    "start"
                ]

                end = window[
                    "end"
                ]

                print()
                print(
                    f"[{window_index}/"
                    f"{len(windows)}] "
                    f"{start:.1f}s → "
                    f"{end:.1f}s"
                )

                start_time = (
                    time.time()
                )

                try:

                    prediction = (
                        analyze_window(
                            model=model,
                            processor=processor,
                            video_path=video_path,
                            start=start,
                            end=end,
                        )
                    )

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    predicted = (
                        prediction[
                            "classification"
                        ]
                    )

                    print(
                        f"Prediction: "
                        f"{predicted}"
                    )

                    print(
                        f"Evidence: "
                        f"{prediction['evidence']}"
                    )

                    print(
                        f"Time: "
                        f"{elapsed:.2f}s"
                    )

                    results.append({

                        "video_path":
                            str(
                                video_path
                            ),

                        "source_label":
                            source_label,

                        "start":
                            start,

                        "end":
                            end,

                        "classification":
                            predicted,

                        "evidence":
                            prediction[
                                "evidence"
                            ],

                        "incident_summary":
                            prediction[
                                "incident_summary"
                            ],

                        "processing_time":
                            round(
                                elapsed,
                                2
                            ),

                    })

                except Exception as error:

                    print(
                        f"ERROR: "
                        f"{error}"
                    )

                    results.append({

                        "video_path":
                            str(
                                video_path
                            ),

                        "source_label":
                            source_label,

                        "start":
                            start,

                        "end":
                            end,

                        "classification":
                            "ERROR",

                        "evidence":
                            "",

                        "incident_summary":
                            "",

                        "error":
                            str(error),

                    })

                finally:

                    cleanup_gpu()

    finally:

        # ====================================================
        # RELEASE QWEN
        # ====================================================

        try:

            del model

        except Exception:

            pass

        try:

            del processor

        except Exception:

            pass

        cleanup_gpu()

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_ROOT
        / "pilot_temporal_labels.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    counts = {

        "Normal": 0,

        "Fire": 0,

        "Fight": 0,

        "Road Accident": 0,

        "ERROR": 0,
    }

    for result in results:

        label = result[
            "classification"
        ]

        if label not in counts:

            counts[label] = 0

        counts[label] += 1

    print()
    print("=" * 70)
    print(
        "PILOT COMPLETE"
    )
    print("=" * 70)

    print(
        f"Total windows analyzed: "
        f"{len(results)}"
    )

    print()

    print(
        "Predicted window counts:"
    )

    for label, count in counts.items():

        print(
            f"  {label:<15}"
            f"{count}"
        )

    print()

    print(
        f"Saved:"
    )

    print(
        f"{output_path}"
    )

    print()
    print(
        "DO NOT START LoRA TRAINING YET."
    )

    print(
        "Review the temporal predictions first."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()