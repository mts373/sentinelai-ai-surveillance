import json
import random
import cv2
from pathlib import Path
from collections import Counter


# ============================================================
# SENTINELAI - CREATE TEMPORAL QWEN2.5-VL SFT DATASET
# ============================================================
#
# INPUT:
#
#   dataset\sft_annotations\train_annotations.json
#
# OUTPUT:
#
#   dataset\sft\clips\
#       actual 10-second training clips
#
#   dataset\sft\sft_train.json
#
#   dataset\sft\sft_summary.json
#
#
# IMPORTANT:
#
# The original UCF video is NOT passed directly to SFT.
#
# Instead:
#
# Original video
#       ↓
# Human-verified temporal window
#       ↓
# Extract exact clip
#       ↓
# Qwen2.5-VL SFT
#
# This prevents the model from seeing unrelated parts of the
# source video while learning the temporal label.
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

ANNOTATION_FILE = (
    DATASET_ROOT
    / "sft_annotations"
    / "train_annotations.json"
)

OUTPUT_ROOT = (
    DATASET_ROOT
    / "sft"
)

CLIPS_ROOT = (
    OUTPUT_ROOT
    / "clips"
)

OUTPUT_FILE = (
    OUTPUT_ROOT
    / "sft_train.json"
)

SUMMARY_FILE = (
    OUTPUT_ROOT
    / "sft_summary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42


TARGET_COUNTS = {

    "Normal": 60,

    "Fire": 10,

    "Fight": 15,

    "Road Accident": 23,

}


CLASS_ORDER = [

    "Normal",

    "Fire",

    "Fight",

    "Road Accident",

]


# ============================================================
# LOAD JSON
# ============================================================

def load_json(
    path
):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n"
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
# VALIDATE ANNOTATIONS
# ============================================================

def validate_annotations(
    annotations
):

    valid = []

    invalid = []

    for index, item in enumerate(
        annotations
    ):

        required = [

            "clip_id",

            "video_path",

            "actual_label",

            "start",

            "end",

        ]

        missing = [

            field

            for field in required

            if field not in item

        ]

        if missing:

            invalid.append(
                (
                    index,
                    f"Missing: {missing}"
                )
            )

            continue

        label = item[
            "actual_label"
        ]

        if label not in CLASS_ORDER:

            invalid.append(
                (
                    index,
                    f"Invalid label: {label}"
                )
            )

            continue

        video_path = Path(
            item[
                "video_path"
            ]
        )

        if not video_path.exists():

            invalid.append(
                (
                    index,
                    "Video file does not exist"
                )
            )

            continue

        try:

            start = float(
                item[
                    "start"
                ]
            )

            end = float(
                item[
                    "end"
                ]
            )

        except (
            TypeError,
            ValueError,
        ):

            invalid.append(
                (
                    index,
                    "Invalid start/end"
                )
            )

            continue

        if start < 0:

            invalid.append(
                (
                    index,
                    "Start time is negative"
                )
            )

            continue

        if end <= start:

            invalid.append(
                (
                    index,
                    "End must be greater than start"
                )
            )

            continue

        valid.append(
            item
        )

    return valid, invalid


# ============================================================
# GROUP BY CLASS
# ============================================================

def group_by_class(
    annotations
):

    grouped = {

        label: []

        for label in CLASS_ORDER

    }

    for item in annotations:

        grouped[
            item[
                "actual_label"
            ]
        ].append(
            item
        )

    return grouped


# ============================================================
# SELECT SUBSET
# ============================================================

def select_subset(
    grouped
):

    random.seed(
        RANDOM_SEED
    )

    selected = []

    print()
    print("=" * 70)

    print(
        "SELECTING VERIFIED SFT DATA"
    )

    print("=" * 70)

    for label in CLASS_ORDER:

        available = grouped[
            label
        ]

        target = TARGET_COUNTS[
            label
        ]

        if len(available) < target:

            raise ValueError(
                f"Not enough verified "
                f"{label} examples.\n"
                f"Required: {target}\n"
                f"Available: {len(available)}"
            )

        candidates = (
            available.copy()
        )

        random.shuffle(
            candidates
        )

        chosen = candidates[
            :target
        ]

        selected.extend(
            chosen
        )

        print(
            f"{label:<15}"
            f"Available: "
            f"{len(available):<4}"
            f"Selected: "
            f"{len(chosen)}"
        )

    random.shuffle(
        selected
    )

    return selected


# ============================================================
# EXTRACT EXACT TEMPORAL CLIP
# ============================================================

def extract_temporal_clip(
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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    # --------------------------------------------------------
    # Clamp temporal boundaries
    # --------------------------------------------------------

    start = max(
        0.0,
        float(start)
    )

    end = min(
        float(end),
        duration
    )

    if end <= start:

        cap.release()

        raise ValueError(
            f"Invalid temporal range "
            f"{start} → {end}"
        )

    start_frame = int(
        round(
            start * fps
        )
    )

    end_frame = int(
        round(
            end * fps
        )
    )

    start_frame = max(
        0,
        min(
            start_frame,
            total_frames - 1
        )
    )

    end_frame = max(
        start_frame + 1,
        min(
            end_frame,
            total_frames
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

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
            f"Could not create output:\n"
            f"{output_path}"
        )

    # --------------------------------------------------------
    # Seek
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    frame_number = (
        start_frame
    )

    written = 0

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

            written += 1

    finally:

        cap.release()

        writer.release()

    if written == 0:

        raise RuntimeError(
            "No frames were written."
        )

    return {

        "fps":
            fps,

        "width":
            width,

        "height":
            height,

        "frames":
            written,

        "duration":
            written / fps,

    }


# ============================================================
# BUILD ASSISTANT TARGET
# ============================================================

def build_assistant_response(
    label
):

    if label == "Normal":

        evidence = (
            "No Fire, Fight, or Road "
            "Accident is visibly present "
            "in this temporal window."
        )

        summary = (
            "The video segment shows "
            "normal activity without a "
            "targeted security incident."
        )

    elif label == "Fire":

        evidence = (
            "Visible fire or flames are "
            "present in the temporal window."
        )

        summary = (
            "The video segment contains "
            "a fire-related incident."
        )

    elif label == "Fight":

        evidence = (
            "People are visibly engaged "
            "in a physical altercation."
        )

        summary = (
            "The video segment contains "
            "a physical fight or violent "
            "altercation."
        )

    elif label == "Road Accident":

        evidence = (
            "A visible road traffic "
            "accident or collision occurs "
            "in the temporal window."
        )

        summary = (
            "The video segment contains "
            "a road traffic accident."
        )

    else:

        raise ValueError(
            f"Unsupported label: {label}"
        )

    target = {

        "classification":
            label,

        "evidence":
            evidence,

        "incident_summary":
            summary,

    }

    return json.dumps(
        target,
        ensure_ascii=False,
    )


# ============================================================
# BUILD QWEN SFT RECORD
# ============================================================

def build_sft_record(
    item,
    clip_path,
    index
):

    label = item[
        "actual_label"
    ]

    assistant_response = (
        build_assistant_response(
            label
        )
    )

    user_text = (
        "Analyze this surveillance "
        "video clip. "
        "Classify the scene as exactly "
        "one of: Normal, Fire, Fight, "
        "or Road Accident. "
        "Return a JSON object with the "
        "fields classification, evidence, "
        "and incident_summary."
    )

    record = {

        "id":
            f"sft_{index:05d}",

        "video":
            str(
                clip_path.resolve()
            ),

        "start":
            0.0,

        "end":
            None,

        "label":
            label,

        "messages": [

            {

                "role":
                    "user",

                "content": [

                    {

                        "type":
                            "video",

                        "video":
                            str(
                                clip_path.resolve()
                            ),

                    },

                    {

                        "type":
                            "text",

                        "text":
                            user_text,

                    },

                ],

            },

            {

                "role":
                    "assistant",

                "content": [

                    {

                        "type":
                            "text",

                        "text":
                            assistant_response,

                    },

                ],

            },

        ],

        "metadata": {

            "clip_id":
                item[
                    "clip_id"
                ],

            "source_video":
                item[
                    "video_path"
                ],

            "original_start":
                float(
                    item[
                        "start"
                    ]
                ),

            "original_end":
                float(
                    item[
                        "end"
                    ]
                ),

            "actual_label":
                label,

            "source_label":
                item.get(
                    "source_label"
                ),

            "qwen_prediction":
                item.get(
                    "qwen_prediction"
                ),

        },

    }

    return record


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "SENTINELAI - TEMPORAL "
        "QWEN2.5-VL SFT DATASET"
    )

    print("=" * 70)

    print()

    print(
        "Human annotation source:"
    )

    print(
        ANNOTATION_FILE
    )

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    annotations = load_json(
        ANNOTATION_FILE
    )

    print()

    print(
        f"Loaded annotations: "
        f"{len(annotations)}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    valid, invalid = (
        validate_annotations(
            annotations
        )
    )

    print(
        f"Valid annotations: "
        f"{len(valid)}"
    )

    print(
        f"Invalid annotations: "
        f"{len(invalid)}"
    )

    if invalid:

        print()

        print(
            "INVALID RECORDS:"
        )

        for index, reason in invalid:

            print(
                f"  [{index}] "
                f"{reason}"
            )

        raise RuntimeError(
            "Fix invalid annotations "
            "before continuing."
        )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    grouped = group_by_class(
        valid
    )

    print()

    print(
        "VERIFIED DISTRIBUTION:"
    )

    for label in CLASS_ORDER:

        print(
            f"  {label:<15}"
            f"{len(grouped[label])}"
        )

    # --------------------------------------------------------
    # Select
    # --------------------------------------------------------

    selected = select_subset(
        grouped
    )

    print()

    print(
        f"Selected examples: "
        f"{len(selected)}"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    CLIPS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Extract clips
    # --------------------------------------------------------

    dataset = []

    print()
    print("=" * 70)

    print(
        "EXTRACTING VERIFIED TEMPORAL CLIPS"
    )

    print("=" * 70)

    print()

    total = len(
        selected
    )

    for index, item in enumerate(
        selected,
        start=1,
    ):

        clip_id = item[
            "clip_id"
        ]

        label = item[
            "actual_label"
        ]

        source_video = Path(
            item[
                "video_path"
            ]
        )

        start = float(
            item[
                "start"
            ]
        )

        end = float(
            item[
                "end"
            ]
        )

        # ----------------------------------------------------
        # Safe filename
        # ----------------------------------------------------

        filename = (
            f"{index:03d}_"
            f"{label.replace(' ', '_')}_"
            f"{clip_id}.mp4"
        )

        clip_path = (
            CLIPS_ROOT
            / filename
        )

        print(
            f"[{index}/{total}] "
            f"{label:<15} "
            f"{source_video.name} "
            f"{start:.2f}s → "
            f"{end:.2f}s"
        )

        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        try:

            info = extract_temporal_clip(
                source_video,
                start,
                end,
                clip_path,
            )

        except Exception as error:

            print(
                f"    ERROR: {error}"
            )

            raise

        print(
            f"    Created: "
            f"{clip_path.name}"
        )

        print(
            f"    Resolution: "
            f"{info['width']}x"
            f"{info['height']}"
        )

        print(
            f"    FPS: "
            f"{info['fps']:.2f}"
        )

        print(
            f"    Frames: "
            f"{info['frames']}"
        )

        # ----------------------------------------------------
        # Build SFT record
        # ----------------------------------------------------

        record = build_sft_record(
            item,
            clip_path,
            index,
        )

        # Store actual extracted duration.

        record[
            "end"
        ] = round(
            info[
                "duration"
            ],
            3
        )

        record[
            "metadata"
        ][
            "extracted_duration"
        ] = round(
            info[
                "duration"
            ],
            3
        )

        dataset.append(
            record
        )

    # --------------------------------------------------------
    # Save SFT JSON
    # --------------------------------------------------------

    save_json(
        OUTPUT_FILE,
        dataset
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    distribution = Counter(
        item[
            "label"
        ]
        for item in dataset
    )

    summary = {

        "source_annotations":
            str(
                ANNOTATION_FILE
            ),

        "total_verified":
            len(valid),

        "total_sft":
            len(dataset),

        "distribution":
            dict(
                distribution
            ),

        "targets":
            TARGET_COUNTS,

        "random_seed":
            RANDOM_SEED,

        "temporal_clips":
            True,

        "original_videos_used_directly":
            False,

        "notes": [

            "Each SFT example uses an "
            "extracted temporal clip.",

            "Human actual_label values are "
            "the training labels.",

            "Skipped clips are excluded.",

            "Qwen predictions are retained "
            "only as metadata.",

            "The original source video is "
            "not passed to SFT.",

            "The temporal window is extracted "
            "before training."

        ],

    }

    save_json(
        SUMMARY_FILE,
        summary
    )

    # --------------------------------------------------------
    # Inspect first examples
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "SFT SAMPLE INSPECTION"
    )

    print("=" * 70)

    for item in dataset[
        :4
    ]:

        print()

        print(
            f"ID: "
            f"{item['id']}"
        )

        print(
            f"Label: "
            f"{item['label']}"
        )

        print(
            f"Training clip:"
        )

        print(
            item[
                "video"
            ]
        )

        print(
            f"Original window: "
            f"{item['metadata']['original_start']:.2f}s "
            f"→ "
            f"{item['metadata']['original_end']:.2f}s"
        )

        print(
            "Assistant target:"
        )

        print(
            item[
                "messages"
            ][1][
                "content"
            ][0][
                "text"
            ]
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "TEMPORAL SFT DATASET CREATED"
    )

    print("=" * 70)

    print()

    print(
        "Final distribution:"
    )

    for label in CLASS_ORDER:

        print(
            f"  {label:<15}"
            f"{distribution[label]}"
        )

    print()

    print(
        f"TOTAL: "
        f"{len(dataset)}"
    )

    print()

    print(
        "Extracted clips:"
    )

    print(
        CLIPS_ROOT
    )

    print()

    print(
        "SFT JSON:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "Summary:"
    )

    print(
        SUMMARY_FILE
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The SFT JSON now points to the "
        "actual 10-second temporal clips."
    )

    print(
        "Do NOT start LoRA training yet."
    )

    print(
        "We will inspect the generated "
        "dataset and one extracted clip first."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()