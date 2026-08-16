import json
from pathlib import Path

import cv2


# ============================================================
# SENTINELAI - WRONG TEST WINDOW EXTRACTOR
# OpenCV version - NO FFMPEG REQUIRED
# ============================================================

PROJECT_ROOT = Path(r"C:\SentinelAI_Qwen")

RESULTS_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "test_evaluation"
    / "test_results.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "test_evaluation"
    / "review_wrong"
)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{RESULTS_FILE}"
        )

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    # Your evaluation file contains window records.
    if isinstance(data, list):

        return data

    if isinstance(data, dict):

        for key in [
            "window_results",
            "windows",
            "results",
            "window_level_results",
        ]:

            value = data.get(key)

            if isinstance(value, list):
                return value

    raise RuntimeError(
        "Could not locate window-level results "
        "inside test_results.json."
    )


# ============================================================
# EXTRACT ONE WINDOW USING OPENCV
# ============================================================

def extract_window(
    source_path,
    start_seconds,
    end_seconds,
    output_path,
):

    cap = cv2.VideoCapture(
        str(source_path)
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video:"
        )

        print(source_path)

        return False

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:

        cap.release()

        print(
            "ERROR: Invalid FPS."
        )

        return False

    duration = (
        total_frames / fps
    )

    # --------------------------------------------------------
    # Clamp requested window to actual video duration
    # --------------------------------------------------------

    start_seconds = max(
        0.0,
        start_seconds,
    )

    end_seconds = min(
        end_seconds,
        duration,
    )

    if end_seconds <= start_seconds:

        cap.release()

        print(
            "ERROR: Invalid time window."
        )

        return False

    start_frame = int(
        start_seconds * fps
    )

    end_frame = int(
        end_seconds * fps
    )

    # --------------------------------------------------------
    # Seek to starting frame
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    # --------------------------------------------------------
    # Video properties
    # --------------------------------------------------------

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

    source_fps = fps

    # --------------------------------------------------------
    # MP4 writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        source_fps,
        (width, height),
    )

    if not writer.isOpened():

        cap.release()

        print(
            "ERROR: Could not create output video:"
        )

        print(output_path)

        return False

    # --------------------------------------------------------
    # Copy frames
    # --------------------------------------------------------

    frames_written = 0

    while True:

        current_frame = int(
            cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        )

        if current_frame >= end_frame:
            break

        ret, frame = cap.read()

        if not ret:
            break

        writer.write(frame)

        frames_written += 1

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    writer.release()
    cap.release()

    if frames_written == 0:

        print(
            "ERROR: No frames were written."
        )

        return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SENTINELAI - WRONG WINDOW EXTRACTION")
    print("OpenCV / NO FFMPEG")
    print("=" * 70)

    print()
    print("Results:")
    print(RESULTS_FILE)

    records = load_results()

    print()
    print(
        f"Total evaluation records: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find misclassified windows
    # --------------------------------------------------------

    wrong_windows = []

    for record in records:

        ground_truth = record.get(
            "ground_truth"
        )

        prediction = record.get(
            "prediction"
        )

        video_path = record.get(
            "video_path"
        )

        start = record.get(
            "start"
        )

        end = record.get(
            "end"
        )

        if (
            ground_truth is None
            or prediction is None
            or video_path is None
            or start is None
            or end is None
        ):
            continue

        if ground_truth != prediction:

            wrong_windows.append(
                record
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("WRONG WINDOWS FOUND")
    print("=" * 70)

    print(
        f"Total wrong windows: "
        f"{len(wrong_windows)}"
    )

    if len(wrong_windows) == 0:

        print()
        print(
            "No misclassified windows found."
        )

        return

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    manifest = []

    for index, record in enumerate(
        wrong_windows,
        start=1,
    ):

        source_video = Path(
            record["video_path"]
        )

        ground_truth = record[
            "ground_truth"
        ]

        prediction = record[
            "prediction"
        ]

        start = float(
            record["start"]
        )

        end = float(
            record["end"]
        )

        clip_id = record.get(
            "clip_id",
            source_video.stem,
        )

        print()
        print("-" * 70)

        print(
            f"[{index}/{len(wrong_windows)}]"
        )

        print(
            f"Video: "
            f"{source_video.name}"
        )

        print(
            f"Ground truth: "
            f"{ground_truth}"
        )

        print(
            f"Prediction: "
            f"{prediction}"
        )

        print(
            f"Window: "
            f"{start:.2f}s → "
            f"{end:.2f}s"
        )

        # ----------------------------------------------------
        # Verify source
        # ----------------------------------------------------

        if not source_video.exists():

            print(
                "SOURCE VIDEO NOT FOUND:"
            )

            print(
                source_video
            )

            continue

        # ----------------------------------------------------
        # Filename
        # ----------------------------------------------------

        safe_truth = (
            ground_truth
            .replace(" ", "_")
        )

        safe_prediction = (
            prediction
            .replace(" ", "_")
        )

        output_name = (
            f"{index:03d}"
            f"_GT_{safe_truth}"
            f"_PRED_{safe_prediction}"
            f"_{source_video.stem}"
            f"_{start:.2f}s"
            f"_{end:.2f}s"
            ".mp4"
        )

        output_path = (
            OUTPUT_DIR
            / output_name
        )

        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        print(
            "Extracting with OpenCV..."
        )

        success = extract_window(
            source_video,
            start,
            end,
            output_path,
        )

        if not success:
            continue

        print(
            f"Saved: "
            f"{output_path.name}"
        )

        manifest.append(
            {
                "review_file": str(
                    output_path
                ),

                "source_video": str(
                    source_video
                ),

                "clip_id": clip_id,

                "start": start,

                "end": end,

                "ground_truth": (
                    ground_truth
                ),

                "prediction": (
                    prediction
                ),

                "raw_output": record.get(
                    "raw_output"
                ),
            }
        )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest_path = (
        OUTPUT_DIR
        / "review_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Wrong windows found: "
        f"{len(wrong_windows)}"
    )

    print(
        f"Successfully extracted: "
        f"{len(manifest)}"
    )

    print()
    print("Review folder:")
    print(OUTPUT_DIR)

    print()
    print("Manifest:")
    print(manifest_path)

    print()
    print(
        "Filename format:"
    )

    print(
        "GROUND TRUTH → PREDICTION → VIDEO → TIME"
    )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()