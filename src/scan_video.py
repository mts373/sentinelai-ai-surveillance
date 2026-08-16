from pathlib import Path
import argparse
import json
import time
import re

import cv2
import torch
from PIL import Image

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

WINDOW_SECONDS = 10

OVERLAP_SECONDS = 5

NUM_FRAMES = 16

MAX_NEW_TOKENS = 256

CLASSES = [
    "Normal",
    "Fire",
    "Fight",
    "Road Accident",
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# QWEN PROMPT
# ============================================================

PROMPT = """
You are SentinelAI, a CCTV security analysis system.

Analyze the provided sequence of frames. These frames come from
ONE continuous 10-second CCTV video window.

You MUST classify the window into exactly ONE of these four classes:

1. Normal
2. Fire
3. Fight
4. Road Accident

Do NOT use any other class.

Definitions:

- Normal:
  No visible fire, physical fight, or road traffic accident.

- Fire:
  Visible fire, flames, or a clearly occurring fire incident.

- Fight:
  Two or more people are physically fighting, attacking,
  or engaged in violent physical confrontation.

- Road Accident:
  A visible road traffic accident involving vehicles,
  pedestrians, collisions, crashes, or similar traffic incidents.

Important:
- Use only visible evidence in the frames.
- Do not infer an incident merely because a scene looks unusual.
- If none of the three incidents is clearly visible, classify as Normal.
- Look across ALL frames before deciding.
- Do not classify based on a single ambiguous frame.

Return ONLY valid JSON:

{
  "classification": "Normal",
  "evidence": "brief visible evidence",
  "incident_summary": "brief summary",
  "threat_level": "LOW"
}

classification MUST be exactly one of:
Normal
Fire
Fight
Road Accident

threat_level MUST be exactly one of:
LOW
MEDIUM
HIGH
"""


# ============================================================
# PARSE QWEN JSON
# ============================================================

def parse_qwen_response(text):

    cleaned = text.strip()

    # Remove markdown code fences if present.
    cleaned = re.sub(
        r"^```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"^```\s*",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    cleaned = cleaned.strip()


    # Try direct JSON parsing.
    try:

        result = json.loads(
            cleaned
        )

    except json.JSONDecodeError:

        # Try to extract the first JSON object.
        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL,
        )

        if match is None:

            return {
                "classification": "Normal",
                "evidence": (
                    "Qwen response could not "
                    "be parsed as JSON."
                ),
                "incident_summary": cleaned,
                "threat_level": "LOW",
                "parse_error": True,
            }

        try:

            result = json.loads(
                match.group(0)
            )

        except json.JSONDecodeError:

            return {
                "classification": "Normal",
                "evidence": (
                    "Qwen response contained "
                    "invalid JSON."
                ),
                "incident_summary": cleaned,
                "threat_level": "LOW",
                "parse_error": True,
            }


    # --------------------------------------------------------
    # Validate classification
    # --------------------------------------------------------

    classification = result.get(
        "classification",
        "Normal",
    )

    if classification not in CLASSES:

        classification = "Normal"


    # --------------------------------------------------------
    # Validate threat level
    # --------------------------------------------------------

    threat_level = result.get(
        "threat_level",
        "LOW",
    )

    if threat_level not in [
        "LOW",
        "MEDIUM",
        "HIGH",
    ]:

        threat_level = "LOW"


    return {

        "classification":
            classification,

        "evidence":
            str(
                result.get(
                    "evidence",
                    "",
                )
            ),

        "incident_summary":
            str(
                result.get(
                    "incident_summary",
                    "",
                )
            ),

        "threat_level":
            threat_level,

        "parse_error":
            False,
    }


# ============================================================
# SAMPLE WINDOW
# ============================================================

def sample_window(
    video_path,
    start_seconds,
    end_seconds,
    num_frames,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )


    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    if fps <= 0:

        cap.release()

        raise RuntimeError(
            "Invalid video FPS."
        )


    duration = (
        total_frames / fps
    )


    # Clamp window.
    start_seconds = max(
        0.0,
        start_seconds,
    )

    end_seconds = min(
        duration,
        end_seconds,
    )


    if end_seconds <= start_seconds:

        cap.release()

        raise RuntimeError(
            "Invalid temporal window."
        )


    start_frame = int(
        start_seconds * fps
    )

    end_frame = int(
        end_seconds * fps
    )


    if end_frame <= start_frame:

        cap.release()

        raise RuntimeError(
            "Temporal window contains "
            "no frames."
        )


    frame_indices = [
        int(x)
        for x in torch.linspace(
            start_frame,
            end_frame - 1,
            num_frames,
        ).tolist()
    ]


    frames = []


    for frame_index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        success, frame = cap.read()


        if not success:

            cap.release()

            raise RuntimeError(
                f"Failed reading frame "
                f"{frame_index}."
            )


        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )


        frames.append(
            Image.fromarray(
                frame
            )
        )


    cap.release()


    return frames, duration


# ============================================================
# QWEN WINDOW INFERENCE
# ============================================================

def analyze_window(
    model,
    processor,
    frames,
):

    content = [
        {
            "type": "text",
            "text": PROMPT,
        }
    ]


    for frame in frames:

        content.append(
            {
                "type": "image",
                "image": frame,
            }
        )


    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]


    text = (
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


    inputs = processor(
        text=[text],
        images=frames,
        padding=True,
        return_tensors="pt",
    )


    inputs = {
        key: value.to(DEVICE)
        if torch.is_tensor(value)
        else value
        for key, value in inputs.items()
    }


    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )


    generated_ids_trimmed = [

        output_ids[
            len(input_ids):
        ]

        for input_ids, output_ids
        in zip(
            inputs["input_ids"],
            generated_ids,
        )
    ]


    output_text = (
        processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    )


    return (
        parse_qwen_response(
            output_text
        ),
        output_text,
    )


# ============================================================
# FINAL VIDEO DECISION
# ============================================================

def aggregate_results(
    window_results,
):

    # --------------------------------------------------------
    # Count classifications
    # --------------------------------------------------------

    counts = {
        class_name: 0
        for class_name in CLASSES
    }


    for result in window_results:

        classification = (
            result["classification"]
        )

        if classification in counts:

            counts[classification] += 1


    # --------------------------------------------------------
    # Incident priority
    #
    # We don't want Normal to win simply because there
    # are many normal windows in a long video.
    # --------------------------------------------------------

    incident_classes = [
        "Fire",
        "Fight",
        "Road Accident",
    ]


    incident_scores = {}


    for class_name in incident_classes:

        matching = [

            result
            for result in window_results

            if result["classification"]
            == class_name

        ]


        # Number of supporting windows.
        count = len(
            matching
        )


        # Threat-weighted support.
        threat_score = 0

        for result in matching:

            threat = result[
                "threat_level"
            ]

            if threat == "HIGH":

                threat_score += 3

            elif threat == "MEDIUM":

                threat_score += 2

            else:

                threat_score += 1


        incident_scores[
            class_name
        ] = (
            count,
            threat_score,
        )


    # --------------------------------------------------------
    # Select strongest incident.
    # --------------------------------------------------------

    best_class = max(
        incident_classes,
        key=lambda class_name:
            (
                incident_scores[
                    class_name
                ][0],

                incident_scores[
                    class_name
                ][1],
            ),
    )


    best_count = incident_scores[
        best_class
    ][0]


    # If no incident window exists.
    if best_count == 0:

        final_classification = (
            "Normal"
        )

    else:

        final_classification = (
            best_class
        )


    # --------------------------------------------------------
    # Incident windows
    # --------------------------------------------------------

    incident_windows = [

        result
        for result in window_results

        if result["classification"]
        == final_classification

    ]


    return {

        "final_classification":
            final_classification,

        "window_counts":
            counts,

        "incident_scores":
            incident_scores,

        "incident_windows":
            incident_windows,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Scan an entire CCTV video "
            "using Qwen2.5-VL."
        )
    )


    parser.add_argument(
        "video",
        type=str,
        help="Path to MP4 video.",
    )


    parser.add_argument(
        "--window",
        type=float,
        default=WINDOW_SECONDS,
        help=(
            "Window length in seconds."
        ),
    )


    parser.add_argument(
        "--overlap",
        type=float,
        default=OVERLAP_SECONDS,
        help=(
            "Overlap between windows."
        ),
    )


    parser.add_argument(
        "--frames",
        type=int,
        default=NUM_FRAMES,
        help=(
            "Frames sampled per window."
        ),
    )


    args = parser.parse_args()


    video_path = Path(
        args.video
    )


    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 70)

    print(
        "SENTINELAI - FULL VIDEO QWEN SCANNER"
    )

    print("=" * 70)


    print()

    print(
        "Device:",
        DEVICE,
    )


    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )


    # ========================================================
    # CHECK VIDEO
    # ========================================================

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{video_path}"
        )


    # ========================================================
    # VIDEO INFO
    # ========================================================

    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )


    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()


    if fps <= 0:

        raise RuntimeError(
            "Invalid video FPS."
        )


    duration = (
        total_frames / fps
    )


    print()

    print(
        "Video:",
        video_path,
    )

    print(
        "FPS:",
        fps,
    )

    print(
        "Frames:",
        total_frames,
    )

    print(
        "Duration:",
        f"{duration:.2f} seconds",
    )


    # ========================================================
    # WINDOW CONFIGURATION
    # ========================================================

    window_seconds = args.window

    overlap_seconds = args.overlap

    step_seconds = (
        window_seconds
        - overlap_seconds
    )


    if step_seconds <= 0:

        raise ValueError(
            "Overlap must be smaller "
            "than window duration."
        )


    # ========================================================
    # CREATE WINDOWS
    # ========================================================

    windows = []

    start = 0.0


    while start < duration:

        end = min(
            start
            + window_seconds,
            duration,
        )


        windows.append(
            (
                start,
                end,
            )
        )


        if end >= duration:

            break


        start += step_seconds


    print()

    print("=" * 70)

    print(
        "TEMPORAL SCANNING"
    )

    print("=" * 70)

    print()

    print(
        "Window:",
        f"{window_seconds:.1f}s",
    )

    print(
        "Overlap:",
        f"{overlap_seconds:.1f}s",
    )

    print(
        "Step:",
        f"{step_seconds:.1f}s",
    )

    print(
        "Total windows:",
        len(windows),
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()

    print("=" * 70)

    print(
        "LOADING QWEN2.5-VL"
    )

    print("=" * 70)


    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    )


    processor = (
        AutoProcessor.from_pretrained(
            MODEL_NAME
        )
    )


    print()

    print(
        "✅ Qwen loaded."
    )


    # ========================================================
    # PROCESS WINDOWS
    # ========================================================

    window_results = []

    total_start_time = (
        time.time()
    )


    for index, (
        start_seconds,
        end_seconds,
    ) in enumerate(
        windows,
        start=1,
    ):


        print()

        print(
            f"[{index}/{len(windows)}] "
            f"{start_seconds:.1f}s → "
            f"{end_seconds:.1f}s"
        )


        window_start_time = (
            time.time()
        )


        try:

            frames, actual_duration = (
                sample_window(
                    video_path,
                    start_seconds,
                    end_seconds,
                    args.frames,
                )
            )


            result, raw_output = (
                analyze_window(
                    model,
                    processor,
                    frames,
                )
            )


        except Exception as error:

            print(
                "❌ Window failed:",
                error,
            )


            result = {

                "classification":
                    "Normal",

                "evidence":
                    "Window processing failed.",

                "incident_summary":
                    str(error),

                "threat_level":
                    "LOW",

                "parse_error":
                    True,
            }


            raw_output = str(
                error
            )


        elapsed = (
            time.time()
            - window_start_time
        )


        window_record = {

            "window_index":
                index,

            "start_seconds":
                start_seconds,

            "end_seconds":
                end_seconds,

            "classification":
                result[
                    "classification"
                ],

            "threat_level":
                result[
                    "threat_level"
                ],

            "evidence":
                result[
                    "evidence"
                ],

            "incident_summary":
                result[
                    "incident_summary"
                ],

            "parse_error":
                result[
                    "parse_error"
                ],

            "raw_output":
                raw_output,

            "inference_time_seconds":
                elapsed,
        }


        window_results.append(
            window_record
        )


        print(
            "Prediction:",
            result[
                "classification"
            ],
        )

        print(
            "Threat:",
            result[
                "threat_level"
            ],
        )

        print(
            "Time:",
            f"{elapsed:.2f}s",
        )


    # ========================================================
    # AGGREGATION
    # ========================================================

    final_result = (
        aggregate_results(
            window_results
        )
    )


    total_time = (
        time.time()
        - total_start_time
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()

    print("=" * 70)

    print(
        "FINAL VIDEO RESULT"
    )

    print("=" * 70)

    print()

    print(
        "Classification:",
        final_result[
            "final_classification"
        ],
    )


    print()

    print(
        "Window counts:"
    )


    for class_name in CLASSES:

        print(
            f"  {class_name:<15}"
            f"{final_result['window_counts'][class_name]}"
        )


    print()

    print(
        "Total processing time:",
        f"{total_time:.2f}s",
    )


    # ========================================================
    # RESULT OBJECT
    # ========================================================

    output = {

        "video":
            str(video_path),

        "model":
            MODEL_NAME,

        "window_seconds":
            window_seconds,

        "overlap_seconds":
            overlap_seconds,

        "frames_per_window":
            args.frames,

        "duration_seconds":
            duration,

        "total_windows":
            len(windows),

        "total_processing_time_seconds":
            total_time,

        "final_classification":
            final_result[
                "final_classification"
            ],

        "window_counts":
            final_result[
                "window_counts"
            ],

        "incident_scores":
            {
                key: list(value)
                for key, value
                in final_result[
                    "incident_scores"
                ].items()
            },

        "incident_windows":
            final_result[
                "incident_windows"
            ],

        "all_windows":
            window_results,
    }


    # ========================================================
    # SAVE
    # ========================================================

    results_dir = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "results"
    )


    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    output_path = (
        results_dir
        / (
            video_path.stem
            + "_scan.json"
        )
    )


    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )


    print()

    print(
        "Result saved:",
        output_path,
    )


    print()

    print("=" * 70)

    print(
        "✅ FULL VIDEO SCAN COMPLETE"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()