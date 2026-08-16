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

WINDOW_SECONDS = 10.0
OVERLAP_SECONDS = 5.0
NUM_FRAMES = 16
MAX_NEW_TOKENS = 300

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
# EVIDENCE-BASED PROMPT
# ============================================================

PROMPT = """
You are SentinelAI, a CCTV security analysis system.

You are analyzing ONE continuous 10-second CCTV video segment.

Your task is to independently determine whether there is visible
evidence of any of these three incidents:

1. FIRE
2. FIGHT
3. ROAD ACCIDENT

IMPORTANT RULES:

- Examine ALL provided frames before deciding.
- Use ONLY visible evidence.
- Do not guess from context.
- Do not assume an incident because a person is running,
  standing, walking, or interacting with objects.
- Do not classify ordinary activity as an incident.
- Do not use information outside the provided frames.

FIRE:
Mark detected=true only when visible flames, active fire,
or another clearly visible fire event is present.

FIGHT:
Mark detected=true only when people are visibly involved
in physical fighting, attacking, hitting, or violent physical
confrontation.

ROAD ACCIDENT:
Mark detected=true only when a vehicle collision, crash,
vehicle-pedestrian accident, or clearly occurring road traffic
accident is visible.

A strange-looking scene is NOT sufficient evidence.

Return ONLY valid JSON with exactly this structure:

{
  "fire": {
    "detected": false,
    "evidence": "brief evidence"
  },
  "fight": {
    "detected": false,
    "evidence": "brief evidence"
  },
  "road_accident": {
    "detected": false,
    "evidence": "brief evidence"
  }
}

The value of "detected" MUST be either true or false.

If an incident is not clearly visible, use false.
"""


# ============================================================
# JSON PARSER
# ============================================================

def parse_response(text):

    cleaned = text.strip()

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


    try:

        data = json.loads(cleaned)

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL,
        )

        if match is None:

            return {
                "fire": {
                    "detected": False,
                    "evidence": "Invalid Qwen response.",
                },
                "fight": {
                    "detected": False,
                    "evidence": "Invalid Qwen response.",
                },
                "road_accident": {
                    "detected": False,
                    "evidence": "Invalid Qwen response.",
                },
                "parse_error": True,
                "raw_output": text,
            }

        try:

            data = json.loads(
                match.group(0)
            )

        except json.JSONDecodeError:

            return {
                "fire": {
                    "detected": False,
                    "evidence": "Invalid JSON.",
                },
                "fight": {
                    "detected": False,
                    "evidence": "Invalid JSON.",
                },
                "road_accident": {
                    "detected": False,
                    "evidence": "Invalid JSON.",
                },
                "parse_error": True,
                "raw_output": text,
            }


    # --------------------------------------------------------
    # Validate each detector
    # --------------------------------------------------------

    result = {}

    for key in [
        "fire",
        "fight",
        "road_accident",
    ]:

        item = data.get(
            key,
            {}
        )

        detected = item.get(
            "detected",
            False
        )

        if not isinstance(
            detected,
            bool
        ):

            detected = (
                str(detected).lower()
                == "true"
            )


        evidence = str(
            item.get(
                "evidence",
                ""
            )
        )


        result[key] = {
            "detected": detected,
            "evidence": evidence,
        }


    result["parse_error"] = False
    result["raw_output"] = text

    return result


# ============================================================
# SAMPLE TEMPORAL WINDOW
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


    start_seconds = max(
        0.0,
        start_seconds
    )

    end_seconds = min(
        duration,
        end_seconds
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
            "Invalid temporal window."
        )


    indices = [
        int(x)
        for x in torch.linspace(
            start_frame,
            end_frame - 1,
            num_frames
        ).tolist()
    ]


    frames = []


    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            index
        )

        success, frame = cap.read()


        if not success:

            cap.release()

            raise RuntimeError(
                f"Failed to read frame "
                f"{index}"
            )


        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        frames.append(
            Image.fromarray(
                frame
            )
        )


    cap.release()

    return frames, duration


# ============================================================
# QWEN INFERENCE
# ============================================================

def analyze_window(
    model,
    processor,
    frames,
):

    content = [
        {
            "type": "text",
            "text": PROMPT
        }
    ]


    for frame in frames:

        content.append(
            {
                "type": "image",
                "image": frame
            }
        )


    messages = [
        {
            "role": "user",
            "content": content
        }
    ]


    text = (
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    )


    inputs = processor(
        text=[text],
        images=frames,
        padding=True,
        return_tensors="pt"
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
            do_sample=False
        )


    generated_ids_trimmed = [
        output_ids[
            len(input_ids):
        ]

        for input_ids, output_ids
        in zip(
            inputs["input_ids"],
            generated_ids
        )
    ]


    output_text = (
        processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    )


    return parse_response(
        output_text
    )


# ============================================================
# WINDOW CLASSIFICATION
# ============================================================

def classify_window(result):

    fire = result[
        "fire"
    ]["detected"]

    fight = result[
        "fight"
    ]["detected"]

    accident = result[
        "road_accident"
    ]["detected"]


    # --------------------------------------------------------
    # Exactly one incident
    # --------------------------------------------------------

    detected = []


    if fire:
        detected.append("Fire")

    if fight:
        detected.append("Fight")

    if accident:
        detected.append(
            "Road Accident"
        )


    # --------------------------------------------------------
    # No incident
    # --------------------------------------------------------

    if len(detected) == 0:

        return "Normal"


    # --------------------------------------------------------
    # One incident
    # --------------------------------------------------------

    if len(detected) == 1:

        return detected[0]


    # --------------------------------------------------------
    # Multiple detections
    #
    # We do NOT silently choose one.
    # This is important for evaluation.
    # --------------------------------------------------------

    return "Multiple"


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video",
        type=str
    )

    parser.add_argument(
        "--window",
        type=float,
        default=WINDOW_SECONDS
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=OVERLAP_SECONDS
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=NUM_FRAMES
    )


    args = parser.parse_args()


    video_path = Path(
        args.video
    )


    print("=" * 70)

    print(
        "SENTINELAI - QWEN EVIDENCE-BASED VIDEO SCANNER"
    )

    print("=" * 70)

    print()

    print(
        "Device:",
        DEVICE
    )


    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # ========================================================
    # VIDEO INFO
    # ========================================================

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{video_path}"
        )


    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open:\n"
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


    duration = (
        total_frames / fps
    )


    print()

    print(
        "Video:",
        video_path
    )

    print(
        "Duration:",
        f"{duration:.2f}s"
    )

    print(
        "FPS:",
        fps
    )


    # ========================================================
    # WINDOWS
    # ========================================================

    step = (
        args.window
        - args.overlap
    )


    if step <= 0:

        raise ValueError(
            "Overlap must be smaller "
            "than window size."
        )


    windows = []

    start = 0.0


    while start < duration:

        end = min(
            start + args.window,
            duration
        )


        windows.append(
            (start, end)
        )


        if end >= duration:

            break


        start += step


    print()

    print(
        "Window:",
        args.window,
        "seconds"
    )

    print(
        "Overlap:",
        args.overlap,
        "seconds"
    )

    print(
        "Total windows:",
        len(windows)
    )


    # ========================================================
    # LOAD QWEN
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
            device_map="auto"
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
    # SCAN
    # ========================================================

    all_results = []

    scan_start = time.time()


    for index, (
        start_seconds,
        end_seconds
    ) in enumerate(
        windows,
        start=1
    ):

        print()

        print(
            f"[{index}/{len(windows)}] "
            f"{start_seconds:.1f}s → "
            f"{end_seconds:.1f}s"
        )


        window_start = time.time()


        try:

            frames, _ = (
                sample_window(
                    video_path,
                    start_seconds,
                    end_seconds,
                    args.frames
                )
            )


            result = (
                analyze_window(
                    model,
                    processor,
                    frames
                )
            )


            classification = (
                classify_window(
                    result
                )
            )


        except Exception as error:

            print(
                "❌ Window failed:",
                error
            )


            result = {

                "fire": {
                    "detected": False,
                    "evidence": str(error)
                },

                "fight": {
                    "detected": False,
                    "evidence": str(error)
                },

                "road_accident": {
                    "detected": False,
                    "evidence": str(error)
                },

                "parse_error": True,

                "raw_output": str(error)
            }


            classification = "Normal"


        elapsed = (
            time.time()
            - window_start
        )


        record = {

            "window_index":
                index,

            "start_seconds":
                start_seconds,

            "end_seconds":
                end_seconds,

            "classification":
                classification,

            "fire":
                result["fire"],

            "fight":
                result["fight"],

            "road_accident":
                result["road_accident"],

            "parse_error":
                result["parse_error"],

            "inference_time_seconds":
                elapsed,

            "raw_output":
                result["raw_output"]
        }


        all_results.append(
            record
        )


        print(
            "Classification:",
            classification
        )

        print(
            "Fire:",
            result[
                "fire"
            ]["detected"]
        )

        print(
            "Fight:",
            result[
                "fight"
            ]["detected"]
        )

        print(
            "Road Accident:",
            result[
                "road_accident"
            ]["detected"]
        )

        print(
            "Time:",
            f"{elapsed:.2f}s"
        )


    # ========================================================
    # AGGREGATE
    # ========================================================

    counts = {
        "Normal": 0,
        "Fire": 0,
        "Fight": 0,
        "Road Accident": 0,
        "Multiple": 0
    }


    for result in all_results:

        counts[
            result["classification"]
        ] += 1


    # --------------------------------------------------------
    # Count independent evidence
    # --------------------------------------------------------

    evidence_counts = {

        "Fire": sum(
            r["fire"]["detected"]
            for r in all_results
        ),

        "Fight": sum(
            r["fight"]["detected"]
            for r in all_results
        ),

        "Road Accident": sum(
            r["road_accident"]["detected"]
            for r in all_results
        )
    }


    # --------------------------------------------------------
    # Final decision
    #
    # We use strongest independent evidence.
    # --------------------------------------------------------

    final_candidates = sorted(
        evidence_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )


    strongest_class = (
        final_candidates[0][0]
    )

    strongest_count = (
        final_candidates[0][1]
    )


    if strongest_count == 0:

        final_classification = (
            "Normal"
        )

    else:

        final_classification = (
            strongest_class
        )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    total_time = (
        time.time()
        - scan_start
    )


    print()

    print("=" * 70)

    print(
        "FINAL VIDEO RESULT"
    )

    print("=" * 70)

    print()

    print(
        "Classification:",
        final_classification
    )

    print()

    print(
        "Window classifications:"
    )


    for class_name, count in counts.items():

        print(
            f"  {class_name:<15}"
            f"{count}"
        )


    print()

    print(
        "Independent evidence:"
    )


    for class_name, count in (
        evidence_counts.items()
    ):

        print(
            f"  {class_name:<15}"
            f"{count} windows"
        )


    print()

    print(
        "Total processing time:",
        f"{total_time:.2f}s"
    )


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


    output = {

        "video":
            str(video_path),

        "model":
            MODEL_NAME,

        "window_seconds":
            args.window,

        "overlap_seconds":
            args.overlap,

        "frames_per_window":
            args.frames,

        "duration_seconds":
            duration,

        "total_windows":
            len(windows),

        "final_classification":
            final_classification,

        "window_classifications":
            counts,

        "independent_evidence_counts":
            evidence_counts,

        "total_processing_time_seconds":
            total_time,

        "windows":
            all_results
    }


    output_path = (
        results_dir
        /
        (
            video_path.stem
            +
            "_evidence_scan.json"
        )
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )


    print()

    print(
        "Result saved:",
        output_path
    )


    print()

    print("=" * 70)

    print(
        "✅ EVIDENCE-BASED SCAN COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()