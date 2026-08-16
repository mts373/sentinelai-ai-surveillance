import json
import cv2
import os
from pathlib import Path


# ============================================================
# SENTINELAI - VERIFIED SFT ANNOTATION TOOL
# ============================================================
#
# Reads candidates produced by:
#
#     video_diverse_candidates.json
#
# For every candidate:
#
#   1. Extract the temporal clip
#   2. Open the clip
#   3. Human watches it
#   4. Human assigns the ACTUAL label
#
# Labels:
#
#   1 = Normal
#   2 = Fire
#   3 = Fight
#   4 = Road Accident
#   5 = Skip / unclear
#
# Existing annotations are preserved.
#
# Qwen predictions are displayed only as reference.
# They are NEVER used as labels.
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

ANNOTATION_ROOT = (
    DATASET_ROOT
    / "sft_annotations"
)

CANDIDATE_FILE = (
    ANNOTATION_ROOT
    / "video_diverse_candidates.json"
)

ANNOTATION_FILE = (
    ANNOTATION_ROOT
    / "train_annotations.json"
)

REVIEW_ROOT = (
    ANNOTATION_ROOT
    / "review_clips"
)


# ============================================================
# LABELS
# ============================================================

LABELS = {

    "1": "Normal",

    "2": "Fire",

    "3": "Fight",

    "4": "Road Accident",

}


# ============================================================
# LOAD JSON
# ============================================================

def load_json(
    path
):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
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
# SAVE JSON
# ============================================================

def save_json(
    path,
    data
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
# UNIQUE CLIP KEY
# ============================================================

def clip_key(
    video_path,
    start,
    end
):

    return (

        str(video_path),

        round(
            float(start),
            3
        ),

        round(
            float(end),
            3
        ),

    )


# ============================================================
# EXTRACT TEMPORAL CLIP
# ============================================================

def extract_clip(
    video_path,
    start,
    end,
    output_path,
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
        float(start)
        * fps
    )

    end_frame = int(
        float(end)
        * fps
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = cv2.VideoWriter(
        str(output_path),
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
            f"Could not create clip:\n"
            f"{output_path}"
        )

    frame_number = (
        start_frame
    )

    try:

        while (
            frame_number
            < end_frame
        ):

            success, frame = (
                cap.read()
            )

            if not success:

                break

            writer.write(
                frame
            )

            frame_number += 1

    finally:

        cap.release()

        writer.release()


# ============================================================
# OPEN VIDEO
# ============================================================

def open_video(
    path
):

    os.startfile(
        str(path)
    )


# ============================================================
# GET EXISTING ANNOTATION KEYS
# ============================================================

def get_existing_keys(
    annotations
):

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
# GET USER LABEL
# ============================================================

def ask_label():

    print()
    print(
        "=================================================="
    )

    print(
        "WATCH THE CLIP AND CLASSIFY WHAT IS ACTUALLY"
    )

    print(
        "VISIBLE IN THIS TEMPORAL WINDOW."
    )

    print(
        "=================================================="
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

    print()

    while True:

        choice = input(
            "Your choice: "
        ).strip()

        if choice == "5":

            return None

        if choice in LABELS:

            return LABELS[
                choice
            ]

        print(
            "Invalid choice. "
            "Enter 1, 2, 3, 4, or 5."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "SENTINELAI - VERIFIED SFT "
        "ANNOTATION"
    )

    print("=" * 70)

    print()

    print(
        "Candidate file:"
    )

    print(
        CANDIDATE_FILE
    )

    print()

    print(
        "Existing annotations:"
    )

    print(
        ANNOTATION_FILE
    )

    print()

    # --------------------------------------------------------
    # Load candidates
    # --------------------------------------------------------

    candidates = load_json(
        CANDIDATE_FILE
    )

    print(
        f"Candidate clips: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # Load existing annotations
    # --------------------------------------------------------

    if ANNOTATION_FILE.exists():

        annotations = load_json(
            ANNOTATION_FILE
        )

    else:

        annotations = []

    existing_keys = (
        get_existing_keys(
            annotations
        )
    )

    print(
        f"Existing annotations: "
        f"{len(annotations)}"
    )

    print()

    # --------------------------------------------------------
    # Remove candidates already annotated
    # --------------------------------------------------------

    remaining = []

    for candidate in candidates:

        key = clip_key(
            candidate[
                "video_path"
            ],
            candidate[
                "start"
            ],
            candidate[
                "end"
            ],
        )

        if key in existing_keys:

            continue

        remaining.append(
            candidate
        )

    print(
        f"New candidates to review: "
        f"{len(remaining)}"
    )

    print()

    if not remaining:

        print(
            "No new candidates remain."
        )

        return

    # --------------------------------------------------------
    # Review directory
    # --------------------------------------------------------

    REVIEW_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Review loop
    # --------------------------------------------------------

    total = len(
        remaining
    )

    for index, candidate in enumerate(
        remaining,
        start=1,
    ):

        video_path = Path(
            candidate[
                "video_path"
            ]
        )

        clip_id = candidate[
            "clip_id"
        ]

        start = candidate[
            "start"
        ]

        end = candidate[
            "end"
        ]

        source_label = candidate[
            "source_label"
        ]

        qwen_prediction = candidate.get(
            "qwen_prediction"
        )

        qwen_evidence = candidate.get(
            "qwen_evidence"
        )

        # ----------------------------------------------------
        # Output clip
        # ----------------------------------------------------

        clip_filename = (
            f"{index:03d}_"
            f"{clip_id}.mp4"
        )

        clip_path = (
            REVIEW_ROOT
            / clip_filename
        )

        print()
        print("=" * 70)

        print(
            f"CLIP [{index}/{total}]"
        )

        print("=" * 70)

        print()

        print(
            f"Source video:"
        )

        print(
            video_path.name
        )

        print()

        print(
            f"Source class:"
        )

        print(
            source_label
        )

        print()

        print(
            f"Temporal window:"
        )

        print(
            f"{start:.2f}s → "
            f"{end:.2f}s"
        )

        print()

        print(
            f"Qwen prediction:"
        )

        print(
            qwen_prediction
            if qwen_prediction
            else "Unknown"
        )

        if qwen_evidence:

            print()

            print(
                "Qwen evidence:"
            )

            print(
                qwen_evidence
            )

        print()

        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        if not clip_path.exists():

            print(
                "Creating review clip..."
            )

            try:

                extract_clip(
                    video_path,
                    start,
                    end,
                    clip_path,
                )

            except Exception as error:

                print()

                print(
                    "ERROR creating clip:"
                )

                print(
                    error
                )

                print()

                print(
                    "Skipping this candidate."
                )

                continue

        # ----------------------------------------------------
        # Open
        # ----------------------------------------------------

        print(
            "Opening video..."
        )

        open_video(
            clip_path
        )

        # ----------------------------------------------------
        # Ask human
        # ----------------------------------------------------

        actual_label = ask_label()

        # ----------------------------------------------------
        # Skip
        # ----------------------------------------------------

        if actual_label is None:

            print()

            print(
                "SKIPPED."
            )

            continue

        # ----------------------------------------------------
        # Save verified annotation
        # ----------------------------------------------------

        annotation = {

            "clip_id":
                clip_id,

            "video_path":
                str(
                    video_path
                ),

            "source_label":
                source_label,

            "actual_label":
                actual_label,

            "start":
                start,

            "end":
                end,

            "duration":
                candidate[
                    "duration"
                ],

            "split":
                candidate[
                    "split"
                ],

            "qwen_prediction":
                qwen_prediction,

            "qwen_evidence":
                qwen_evidence,

        }

        annotations.append(
            annotation
        )

        existing_keys.add(
            clip_key(
                video_path,
                start,
                end,
            )
        )

        # ----------------------------------------------------
        # Save immediately
        # ----------------------------------------------------

        save_json(
            ANNOTATION_FILE,
            annotations
        )

        print()

        print(
            "VERIFIED ANNOTATION SAVED"
        )

        print(
            f"Actual label: "
            f"{actual_label}"
        )

        print(
            f"Total saved annotations: "
            f"{len(annotations)}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)

    print(
        "ANNOTATION SESSION COMPLETE"
    )

    print("=" * 70)

    counts = {

        "Normal": 0,

        "Fire": 0,

        "Fight": 0,

        "Road Accident": 0,

    }

    for item in annotations:

        label = item.get(
            "actual_label"
        )

        if label in counts:

            counts[
                label
            ] += 1

    print()

    print(
        "VERIFIED DATASET TOTALS:"
    )

    print()

    for label in [
        "Normal",
        "Fire",
        "Fight",
        "Road Accident",
    ]:

        print(
            f"  {label:<15}"
            f"{counts[label]}"
        )

    print()

    print(
        f"Total verified: "
        f"{len(annotations)}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        ANNOTATION_FILE
    )

    print()

    print(
        "Review clips:"
    )

    print(
        REVIEW_ROOT
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Only YOUR actual_label is used "
        "as the verified label."
    )

    print(
        "Qwen predictions are retained only "
        "for analysis."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()