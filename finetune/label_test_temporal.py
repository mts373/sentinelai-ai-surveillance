import json
import cv2
import random
from pathlib import Path


# ============================================================
# SENTINELAI
# SMART HUMAN TEMPORAL TEST LABELING
#
# Purpose:
#   Create a manageable, human-verified temporal test set.
#
# IMPORTANT:
#   Source labels are used ONLY to allocate sampling effort.
#   They are NEVER used as the human ground-truth label.
#
# Playback:
#   Default = 1.5x
#
# Controls while video is playing:
#   P = pause / resume
#   R = restart current window
#   1 = 1.0x
#   2 = 1.5x
#   3 = 2.0x
#   Q / ESC = finish playback and classify
#
# Classification:
#   1 = Normal
#   2 = Fire
#   3 = Fight
#   4 = Road Accident
#   5 = Skip / unclear
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\SentinelAI_Qwen"
)

TEST_JSON = (
    PROJECT_ROOT
    / "dataset"
    / "test.json"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "dataset"
    / "test_evaluation"
    / "human_temporal_labels.json"
)


# ============================================================
# SAMPLING CONFIGURATION
# ============================================================

# Target number of windows PER VIDEO.
#
# Normal videos:
#   4 windows
#
# Anomaly videos:
#   6 windows
#
# This should keep the total workload around a few hundred
# windows for your 70-video test set.

NORMAL_WINDOWS_PER_VIDEO = 4
ANOMALY_WINDOWS_PER_VIDEO = 6

WINDOW_SECONDS = 10.0

# Deterministic seed.
RANDOM_SEED = 42


# ============================================================
# HUMAN LABELS
# ============================================================

LABELS = {
    "1": "Normal",
    "2": "Fire",
    "3": "Fight",
    "4": "Road Accident",
}


# ============================================================
# LOAD TEST SET
# ============================================================

def load_test_set():

    if not TEST_JSON.exists():

        raise FileNotFoundError(
            f"\nTest dataset not found:\n"
            f"{TEST_JSON}"
        )

    with open(
        TEST_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):

        raise ValueError(
            "test.json must contain a JSON list."
        )

    # Only test records.
    data = [
        item
        for item in data
        if item.get("split") == "test"
    ]

    if not data:

        raise RuntimeError(
            "No records with split='test' "
            "were found in test.json."
        )

    return data


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_duration(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video:\n"
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

        raise RuntimeError(
            f"Invalid FPS for:\n"
            f"{video_path}"
        )

    duration = frames / fps

    return float(duration)


# ============================================================
# CREATE CANDIDATE WINDOWS
# ============================================================

def create_candidate_windows(
    duration,
    target_count,
    seed,
):

    if duration <= 0:

        return []

    # --------------------------------------------------------
    # If video is shorter than one window.
    # --------------------------------------------------------

    if duration <= WINDOW_SECONDS:

        return [
            (
                0.0,
                round(duration, 2),
            )
        ]

    # --------------------------------------------------------
    # Number of possible non-overlapping 10-sec windows.
    # --------------------------------------------------------

    max_start = max(
        0.0,
        duration - WINDOW_SECONDS,
    )

    # --------------------------------------------------------
    # Evenly distribute candidate centers.
    #
    # This gives coverage across the entire video instead of
    # simply taking the first N windows.
    # --------------------------------------------------------

    if target_count == 1:

        starts = [
            max_start / 2.0
        ]

    else:

        starts = []

        for i in range(
            target_count
        ):

            fraction = (
                i
                / (target_count - 1)
            )

            starts.append(
                max_start * fraction
            )

    # --------------------------------------------------------
    # Add a deterministic random component.
    #
    # This prevents every video from always being sampled at
    # exactly the same relative positions.
    # --------------------------------------------------------

    rng = random.Random(seed)

    random_candidates = []

    for _ in range(
        target_count * 3
    ):

        random_candidates.append(
            rng.uniform(
                0.0,
                max_start,
            )
        )

    # --------------------------------------------------------
    # Combine evenly spaced + random candidates.
    # --------------------------------------------------------

    all_starts = (
        starts
        + random_candidates
    )

    # --------------------------------------------------------
    # Remove candidates that are too close to each other.
    # --------------------------------------------------------

    selected = []

    minimum_gap = (
        WINDOW_SECONDS * 0.60
    )

    for start in all_starts:

        if all(
            abs(start - existing)
            >= minimum_gap
            for existing in selected
        ):

            selected.append(
                start
            )

        if len(selected) >= target_count:

            break

    # --------------------------------------------------------
    # If we still don't have enough, fill deterministically.
    # --------------------------------------------------------

    if len(selected) < target_count:

        fallback_count = 100

        for i in range(
            fallback_count
        ):

            start = (
                max_start
                * i
                / max(
                    1,
                    fallback_count - 1,
                )
            )

            if all(
                abs(start - existing)
                >= minimum_gap
                for existing in selected
            ):

                selected.append(
                    start
                )

            if len(selected) >= target_count:

                break

    # --------------------------------------------------------
    # Sort chronologically.
    # --------------------------------------------------------

    selected = sorted(
        selected[:target_count]
    )

    windows = []

    for start in selected:

        end = min(
            start + WINDOW_SECONDS,
            duration,
        )

        if (
            end - start
            >= 2.0
        ):

            windows.append(
                (
                    round(start, 2),
                    round(end, 2),
                )
            )

    return windows


# ============================================================
# PLAY VIDEO WINDOW
# ============================================================

def play_window(
    video_path,
    start,
    end,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video:\n"
            f"{video_path}"
        )

    original_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if original_fps <= 0:

        original_fps = 30.0

    speed = 1.5

    paused = False

    restart_requested = False

    window_name = (
        "SENTINELAI TEMPORAL REVIEW"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        1100,
        700,
    )

    while True:

        if restart_requested:

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                start * 1000.0,
            )

            restart_requested = False

        current_time = (
            cap.get(
                cv2.CAP_PROP_POS_MSEC
            )
            / 1000.0
        )

        if current_time >= end:

            break

        if not paused:

            ret, frame = cap.read()

            if not ret:

                break

            current_time = (
                cap.get(
                    cv2.CAP_PROP_POS_MSEC
                )
                / 1000.0
            )

            # ------------------------------------------------
            # Overlay
            # ------------------------------------------------

            overlay = frame.copy()

            cv2.putText(
                overlay,
                (
                    f"{current_time:.1f}s / "
                    f"{end:.1f}s"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                overlay,
                (
                    f"Speed: {speed:.1f}x"
                ),
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                overlay,
                (
                    "P=Pause  "
                    "R=Restart  "
                    "1=1x  "
                    "2=1.5x  "
                    "3=2x  "
                    "Q=Finish"
                ),
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                window_name,
                overlay,
            )

            # ------------------------------------------------
            # Playback speed.
            #
            # At 1.5x, frame display delay is reduced.
            # ------------------------------------------------

            delay = max(
                1,
                int(
                    1000
                    / (
                        original_fps
                        * speed
                    )
                ),
            )

        else:

            # ------------------------------------------------
            # Pause screen.
            # ------------------------------------------------

            ret, frame = cap.read()

            if not ret:

                break

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                max(
                    0,
                    int(
                        cap.get(
                            cv2.CAP_PROP_POS_FRAMES
                        )
                    ) - 1,
                ),
            )

            cv2.putText(
                frame,
                "PAUSED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3,
            )

            cv2.putText(
                frame,
                (
                    "P=Resume  "
                    "R=Restart  "
                    "1=1x  "
                    "2=1.5x  "
                    "3=2x  "
                    "Q=Finish"
                ),
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                window_name,
                frame,
            )

            delay = 100

        key = cv2.waitKey(
            delay
        ) & 0xFF

        # ----------------------------------------------------
        # Controls
        # ----------------------------------------------------

        if key in (
            ord("p"),
            ord("P"),
        ):

            paused = not paused

        elif key in (
            ord("r"),
            ord("R"),
        ):

            restart_requested = True

            paused = False

        elif key == ord("1"):

            speed = 1.0

        elif key == ord("2"):

            speed = 1.5

        elif key == ord("3"):

            speed = 2.0

        elif key in (
            ord("q"),
            ord("Q"),
            27,
        ):

            break

    cap.release()

    cv2.destroyWindow(
        window_name
    )

    cv2.waitKey(100)


# ============================================================
# HUMAN CLASSIFICATION
# ============================================================

def get_human_label():

    print()
    print(
        "CLASSIFY WHAT IS ACTUALLY VISIBLE:"
    )

    print()
    print(
        "  1 = Normal"
    )
    print(
        "  2 = Fire"
    )
    print(
        "  3 = Fight"
    )
    print(
        "  4 = Road Accident"
    )
    print(
        "  5 = Skip / unclear"
    )

    while True:

        choice = input(
            "\nYour choice: "
        ).strip()

        if choice in LABELS:

            return LABELS[choice]

        if choice == "5":

            return None

        print(
            "Invalid choice."
        )


# ============================================================
# LOAD EXISTING ANNOTATIONS
# ============================================================

def load_existing_annotations():

    if not OUTPUT_JSON.exists():

        return []

    try:

        with open(
            OUTPUT_JSON,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        if isinstance(
            data,
            list,
        ):

            return data

    except Exception as e:

        print()
        print(
            "WARNING: Could not read existing "
            "annotation file."
        )

        print(e)

    return []


# ============================================================
# SAVE
# ============================================================

def save_annotations(
    annotations,
):

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        OUTPUT_JSON.with_suffix(
            ".tmp"
        )
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            annotations,
            f,
            indent=2,
        )

    # Replace only after successful write.
    temporary.replace(
        OUTPUT_JSON
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SENTINELAI - SMART HUMAN TEMPORAL TEST LABELING"
    )
    print("=" * 70)

    print()
    print(
        f"Test set:\n{TEST_JSON}"
    )

    print()
    print(
        f"Output:\n{OUTPUT_JSON}"
    )

    print()
    print(
        "Sampling:"
    )

    print(
        f"Normal videos: "
        f"{NORMAL_WINDOWS_PER_VIDEO} windows/video"
    )

    print(
        f"Anomaly videos: "
        f"{ANOMALY_WINDOWS_PER_VIDEO} windows/video"
    )

    print(
        f"Window duration: "
        f"{WINDOW_SECONDS}s"
    )

    print()
    print(
        "Playback default: 1.5x"
    )

    print(
        "Controls: "
        "P=pause, R=restart, "
        "1=1x, 2=1.5x, 3=2x, Q=finish"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Source labels are ONLY used to "
        "allocate sampling."
    )

    print(
        "They are NOT the human ground truth."
    )

    print(
        "Judge ONLY what is visible."
    )

    # ========================================================
    # DATA
    # ========================================================

    data = load_test_set()

    print()
    print(
        f"Test videos: {len(data)}"
    )

    # ========================================================
    # EXISTING LABELS
    # ========================================================

    annotations = (
        load_existing_annotations()
    )

    completed = set()

    for item in annotations:

        try:

            key = (
                item["video_path"],
                float(item["start"]),
                float(item["end"]),
            )

            completed.add(
                key
            )

        except Exception:

            continue

    print()
    print(
        f"Existing human annotations: "
        f"{len(annotations)}"
    )

    # ========================================================
    # BUILD SAMPLING PLAN
    # ========================================================

    sampling_plan = []

    total_candidates = 0

    for video_index, item in enumerate(
        data,
        start=1,
    ):

        video_path = Path(
            item["video_path"]
        )

        if not video_path.exists():

            print()
            print(
                "WARNING - missing video:"
            )

            print(
                video_path
            )

            continue

        source_label = item.get(
            "label",
            "UNKNOWN",
        )

        # ----------------------------------------------------
        # Normal gets fewer windows.
        # ----------------------------------------------------

        if source_label == "Normal":

            target = (
                NORMAL_WINDOWS_PER_VIDEO
            )

        else:

            target = (
                ANOMALY_WINDOWS_PER_VIDEO
            )

        duration = (
            get_video_duration(
                video_path
            )
        )

        windows = (
            create_candidate_windows(
                duration,
                target,
                RANDOM_SEED
                + video_index,
            )
        )

        for start, end in windows:

            sampling_plan.append(
                {
                    "video_path": str(
                        video_path
                    ),
                    "source_label": (
                        source_label
                    ),
                    "start": start,
                    "end": end,
                    "duration": round(
                        end - start,
                        2,
                    ),
                }
            )

        total_candidates += len(
            windows
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "SAMPLING PLAN"
    )
    print("=" * 70)

    print(
        f"Videos: {len(data)}"
    )

    print(
        f"Candidate windows: "
        f"{total_candidates}"
    )

    print(
        f"Already labeled: "
        f"{len(completed)}"
    )

    remaining = sum(
        1
        for item in sampling_plan
        if (
            item["video_path"],
            item["start"],
            item["end"],
        )
        not in completed
    )

    print(
        f"Remaining to label: "
        f"{remaining}"
    )

    print()
    print(
        "This is intentionally much smaller "
        "than labeling every 5-second window."
    )

    # ========================================================
    # PROCESS
    # ========================================================

    new_labels = 0

    try:

        for index, candidate in enumerate(
            sampling_plan,
            start=1,
        ):

            key = (
                candidate["video_path"],
                candidate["start"],
                candidate["end"],
            )

            if key in completed:

                continue

            video_path = Path(
                candidate["video_path"]
            )

            print()
            print("=" * 70)
            print(
                f"CANDIDATE "
                f"[{index}/{len(sampling_plan)}]"
            )
            print("=" * 70)

            print(
                f"Video: "
                f"{video_path.name}"
            )

            print(
                f"Source dataset class: "
                f"{candidate['source_label']}"
            )

            print(
                f"Window: "
                f"{candidate['start']:.2f}s "
                f"→ "
                f"{candidate['end']:.2f}s"
            )

            print()
            print(
                "Watch the complete window."
            )

            print(
                "Default playback: 1.5x"
            )

            print(
                "Use 1/2/3 to change speed."
            )

            play_window(
                video_path,
                candidate["start"],
                candidate["end"],
            )

            label = get_human_label()

            # ------------------------------------------------
            # Skip
            # ------------------------------------------------

            if label is None:

                print()
                print(
                    "SKIPPED."
                )

                continue

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            record = {
                "video_path": (
                    candidate["video_path"]
                ),
                "source_label": (
                    candidate["source_label"]
                ),
                "start": (
                    candidate["start"]
                ),
                "end": (
                    candidate["end"]
                ),
                "duration": (
                    candidate["duration"]
                ),
                "human_label": label,
            }

            annotations.append(
                record
            )

            completed.add(
                key
            )

            save_annotations(
                annotations
            )

            new_labels += 1

            print()
            print(
                "HUMAN ANNOTATION SAVED"
            )

            print(
                f"Actual label: {label}"
            )

            print(
                f"Total saved: "
                f"{len(annotations)}"
            )

    except KeyboardInterrupt:

        print()
        print()
        print("=" * 70)
        print(
            "LABELING STOPPED BY USER"
        )
        print("=" * 70)

        save_annotations(
            annotations
        )

        print(
            f"Saved annotations: "
            f"{len(annotations)}"
        )

        print(
            f"File:\n{OUTPUT_JSON}"
        )

        print()
        print(
            "Run the same command later "
            "to resume."
        )

        return

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print(
        "SMART TEMPORAL LABELING COMPLETE"
    )
    print("=" * 70)

    print(
        f"Candidate windows: "
        f"{len(sampling_plan)}"
    )

    print(
        f"New human labels: "
        f"{new_labels}"
    )

    print(
        f"Total human annotations: "
        f"{len(annotations)}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_JSON
    )

    print()
    print(
        "DO NOT RETRAIN YET."
    )

    print(
        "Next step is to evaluate the "
        "existing LoRA adapter against "
        "these human temporal labels."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()