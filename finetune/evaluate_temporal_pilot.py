import json
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# SENTINELAI - TEMPORAL PILOT EVALUATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\SentinelAI_Qwen"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "temporal_labels"
    / "pilot_temporal_labels.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "temporal_labels"
    / "pilot_evaluation.json"
)


ANOMALY_CLASSES = [
    "Fire",
    "Fight",
    "Road Accident",
]


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"File not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SENTINELAI - TEMPORAL "
        "PILOT EVALUATION"
    )
    print("=" * 70)

    results = load_results()

    # --------------------------------------------------------
    # Group windows by source video
    # --------------------------------------------------------

    videos = defaultdict(list)

    for result in results:

        videos[
            result["video_path"]
        ].append(result)

    print(
        f"Total videos: {len(videos)}"
    )

    print(
        f"Total windows: {len(results)}"
    )

    print()

    # --------------------------------------------------------
    # Overall counters
    # --------------------------------------------------------

    total_windows = 0

    exact_matches = 0

    anomaly_detected = 0

    anomaly_missed = 0

    false_positive = 0

    video_reports = []

    # --------------------------------------------------------
    # Evaluate every video
    # --------------------------------------------------------

    for video_path, windows in videos.items():

        expected = windows[0][
            "source_label"
        ]

        counts = Counter()

        for window in windows:

            prediction = window[
                "classification"
            ]

            counts[prediction] += 1

            total_windows += 1

            if prediction == expected:

                exact_matches += 1

            # -----------------------------------------------
            # Expected anomaly
            # -----------------------------------------------

            if expected in ANOMALY_CLASSES:

                if prediction == expected:

                    anomaly_detected += 1

                elif prediction == "Normal":

                    anomaly_missed += 1

            # -----------------------------------------------
            # Expected Normal
            # -----------------------------------------------

            elif expected == "Normal":

                if prediction in ANOMALY_CLASSES:

                    false_positive += 1

        detected_windows = 0

        if expected in ANOMALY_CLASSES:

            detected_windows = counts[
                expected
            ]

        # ----------------------------------------------------
        # Video status
        # ----------------------------------------------------

        if expected in ANOMALY_CLASSES:

            if detected_windows > 0:

                status = (
                    "ANOMALY DETECTED"
                )

            else:

                status = (
                    "ANOMALY MISSED"
                )

        else:

            if sum(
                counts[c]
                for c in ANOMALY_CLASSES
            ) == 0:

                status = (
                    "NORMAL CORRECT"
                )

            else:

                status = (
                    "FALSE POSITIVE"
                )

        report = {

            "video_path":
                video_path,

            "expected_class":
                expected,

            "total_windows":
                len(windows),

            "prediction_counts":
                dict(counts),

            "expected_class_windows":
                detected_windows,

            "status":
                status,
        }

        video_reports.append(
            report
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print("-" * 70)

        print(
            f"Video: "
            f"{Path(video_path).name}"
        )

        print(
            f"Expected: "
            f"{expected}"
        )

        print(
            f"Status: "
            f"{status}"
        )

        print(
            "Predictions:"
        )

        for label, count in counts.items():

            print(
                f"  {label:<15}"
                f"{count}"
            )

        # ----------------------------------------------------
        # Show anomaly timestamps
        # ----------------------------------------------------

        if expected in ANOMALY_CLASSES:

            detected = [

                window

                for window in windows

                if window[
                    "classification"
                ] == expected

            ]

            if detected:

                print(
                    "Detected windows:"
                )

                for window in detected:

                    print(
                        f"  "
                        f"{window['start']:.1f}s "
                        f"→ "
                        f"{window['end']:.1f}s"
                    )

            else:

                print(
                    "Detected windows: NONE"
                )

    # ========================================================
    # CLASS-LEVEL SUMMARY
    # ========================================================

    class_summary = {}

    for expected_class in (
        ANOMALY_CLASSES
        + ["Normal"]
    ):

        class_videos = [

            report

            for report in video_reports

            if report[
                "expected_class"
            ] == expected_class
        ]

        if not class_videos:

            continue

        detected_video_count = 0

        missed_video_count = 0

        total_class_windows = 0

        correct_class_windows = 0

        for report in class_videos:

            total_class_windows += (
                report[
                    "total_windows"
                ]
            )

            correct_class_windows += (
                report[
                    "prediction_counts"
                ].get(
                    expected_class,
                    0
                )
            )

            if expected_class in (
                ANOMALY_CLASSES
            ):

                if (
                    report[
                        "expected_class_windows"
                    ] > 0
                ):

                    detected_video_count += 1

                else:

                    missed_video_count += 1

        class_summary[
            expected_class
        ] = {

            "videos":
                len(class_videos),

            "videos_with_expected_class_detected":
                detected_video_count,

            "videos_with_expected_class_missed":
                missed_video_count,

            "total_windows":
                total_class_windows,

            "correct_class_predictions":
                correct_class_windows,
        }

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "CLASS SUMMARY"
    )
    print("=" * 70)

    for label, summary in (
        class_summary.items()
    ):

        print()
        print(
            f"{label}"
        )

        print(
            f"  Videos: "
            f"{summary['videos']}"
        )

        print(
            f"  Windows: "
            f"{summary['total_windows']}"
        )

        print(
            f"  Correct-class windows: "
            f"{summary['correct_class_predictions']}"
        )

        if label in ANOMALY_CLASSES:

            print(
                f"  Videos detected: "
                f"{summary['videos_with_expected_class_detected']}"
            )

            print(
                f"  Videos missed: "
                f"{summary['videos_with_expected_class_missed']}"
            )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    window_accuracy = 0.0

    if total_windows > 0:

        window_accuracy = (
            exact_matches
            / total_windows
        )

    print()
    print("=" * 70)
    print(
        "OVERALL"
    )
    print("=" * 70)

    print(
        f"Total windows: "
        f"{total_windows}"
    )

    print(
        f"Exact window matches: "
        f"{exact_matches}"
    )

    print(
        f"Window accuracy: "
        f"{window_accuracy * 100:.2f}%"
    )

    print(
        f"Anomaly windows detected: "
        f"{anomaly_detected}"
    )

    print(
        f"Anomaly windows missed: "
        f"{anomaly_missed}"
    )

    print(
        f"False-positive windows: "
        f"{false_positive}"
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    output = {

        "input_file":
            str(INPUT_FILE),

        "total_videos":
            len(videos),

        "total_windows":
            total_windows,

        "exact_window_matches":
            exact_matches,

        "window_accuracy":
            round(
                window_accuracy,
                4
            ),

        "anomaly_windows_detected":
            anomaly_detected,

        "anomaly_windows_missed":
            anomaly_missed,

        "false_positive_windows":
            false_positive,

        "class_summary":
            class_summary,

        "videos":
            video_reports,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 70)
    print(
        "EVALUATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()