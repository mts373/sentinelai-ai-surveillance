from pathlib import Path
import argparse
import json
import re
import time

import cv2
import torch
from PIL import Image

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

WINDOW_SECONDS = 10.0
OVERLAP_SECONDS = 5.0
NUM_FRAMES = 16

MAX_NEW_TOKENS_DETECTION = 300
MAX_NEW_TOKENS_SUMMARY = 300

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# DETECTION PROMPT
# ============================================================

DETECTION_PROMPT = """
You are SentinelAI, a CCTV security analysis system.

Analyze ALL frames from this continuous video segment.

Determine whether there is clear visible evidence of any of
these incidents:

1. FIRE
2. FIGHT
3. ROAD ACCIDENT

Rules:

- Use ONLY visible evidence.
- Do not guess.
- Do not classify ordinary walking, running, standing,
  or interacting as an incident.
- FIRE requires visible flames or an active fire.
- FIGHT requires visible physical fighting, hitting,
  attacking, or violent physical confrontation.
- ROAD ACCIDENT requires a visible vehicle collision,
  crash, vehicle-pedestrian accident, or clearly occurring
  road traffic accident.
- If an incident is not clearly visible, use false.

Return ONLY valid JSON:

{
  "fire": {
    "detected": false,
    "evidence": ""
  },
  "fight": {
    "detected": false,
    "evidence": ""
  },
  "road_accident": {
    "detected": false,
    "evidence": ""
  }
}
"""


# ============================================================
# SUMMARY PROMPT
# ============================================================

def build_summary_prompt(
    classification,
    start_seconds,
    end_seconds,
    evidence
):

    return f"""
You are SentinelAI, an intelligent CCTV incident reporting system.

A temporal video analysis system detected:

Incident:
{classification}

Approximate incident time:
{start_seconds:.1f} seconds to {end_seconds:.1f} seconds

Visual evidence:
{evidence}

Generate a concise security incident report.

Return ONLY valid JSON:

{{
  "incident_summary": "brief factual description",
  "threat_level": "LOW",
  "recommended_action": "brief recommended security action"
}}

Threat levels:

LOW:
Minor or non-dangerous situation.

MEDIUM:
Potentially dangerous situation requiring attention.

HIGH:
Clearly dangerous incident requiring immediate attention.

Do not invent details that are not supported by the evidence.
"""


# ============================================================
# JSON PARSER
# ============================================================

def parse_json_response(text):

    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL
        )

        if match:

            try:
                return json.loads(
                    match.group(0)
                )

            except json.JSONDecodeError:
                pass

    return None


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    if fps <= 0:
        raise RuntimeError(
            "Invalid video FPS."
        )

    duration = (
        total_frames / fps
    )

    return fps, total_frames, duration


# ============================================================
# SAMPLE TEMPORAL WINDOW
# ============================================================

def sample_window(
    video_path,
    start_seconds,
    end_seconds
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
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

    indices = torch.linspace(
        start_frame,
        end_frame - 1,
        NUM_FRAMES
    ).long().tolist()

    frames = []

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = cap.read()

        if not success:
            cap.release()
            raise RuntimeError(
                f"Failed to read frame {index}"
            )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(
            Image.fromarray(frame)
        )

    cap.release()

    return frames


# ============================================================
# QWEN CALL
# ============================================================

def qwen_generate(
    model,
    processor,
    frames,
    prompt,
    max_new_tokens
):

    content = [
        {
            "type": "text",
            "text": prompt
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

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
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
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    generated_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(
            inputs["input_ids"],
            generated_ids
        )
    ]

    output = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output


# ============================================================
# PARSE DETECTION
# ============================================================

def parse_detection(text):

    data = parse_json_response(
        text
    )

    default = {
        "fire": {
            "detected": False,
            "evidence": ""
        },
        "fight": {
            "detected": False,
            "evidence": ""
        },
        "road_accident": {
            "detected": False,
            "evidence": ""
        }
    }

    if not isinstance(data, dict):
        return default

    result = {}

    for key in [
        "fire",
        "fight",
        "road_accident"
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

        result[key] = {
            "detected": detected,
            "evidence": str(
                item.get(
                    "evidence",
                    ""
                )
            )
        }

    return result


# ============================================================
# WINDOW CLASSIFICATION
# ============================================================

def classify_detection(
    detection
):

    detected = []

    if detection["fire"]["detected"]:
        detected.append("Fire")

    if detection["fight"]["detected"]:
        detected.append("Fight")

    if detection[
        "road_accident"
    ]["detected"]:

        detected.append(
            "Road Accident"
        )

    if not detected:
        return "Normal"

    if len(detected) == 1:
        return detected[0]

    return "Multiple"


# ============================================================
# BUILD TEMPORAL EPISODES
# ============================================================

def build_episodes(
    incident_windows
):

    if not incident_windows:
        return []

    episodes = []

    current = [
        incident_windows[0]
    ]

    current_end = (
        incident_windows[0]["end"]
    )

    for window in incident_windows[1:]:

        if window["start"] <= (
            current_end + OVERLAP_SECONDS
        ):

            current.append(
                window
            )

            current_end = max(
                current_end,
                window["end"]
            )

        else:

            episodes.append(
                current
            )

            current = [
                window
            ]

            current_end = (
                window["end"]
            )

    episodes.append(
        current
    )

    return episodes


# ============================================================
# CLASSIFY EPISODE
# ============================================================

def classify_episode(
    episode
):

    primary = episode[0]

    primary_class = (
        primary["classification"]
    )

    observations = []

    for item in episode:

        cls = item[
            "classification"
        ]

        if cls not in observations:
            observations.append(cls)

    evidence = []

    for item in episode:

        cls = item[
            "classification"
        ]

        if cls == "Fire":
            text = item[
                "detection"
            ]["fire"]["evidence"]

        elif cls == "Fight":
            text = item[
                "detection"
            ]["fight"]["evidence"]

        elif cls == "Road Accident":
            text = item[
                "detection"
            ]["road_accident"]["evidence"]

        else:
            text = ""

        if text and text not in evidence:
            evidence.append(text)

    return {
        "start_seconds":
            episode[0]["start"],

        "end_seconds":
            max(
                x["end"]
                for x in episode
            ),

        "primary_classification":
            primary_class,

        "observations":
            observations,

        "window_indices":
            [
                x["window_index"]
                for x in episode
            ],

        "evidence":
            evidence
    }


# ============================================================
# INCIDENT SUMMARY
# ============================================================

def generate_incident_report(
    model,
    processor,
    episode
):

    classification = (
        episode[
            "primary_classification"
        ]
    )

    start = (
        episode[
            "start_seconds"
        ]
    )

    end = (
        episode[
            "end_seconds"
        ]
    )

    evidence = "; ".join(
        episode["evidence"]
    )

    prompt = build_summary_prompt(
        classification,
        start,
        end,
        evidence
    )

    # We don't need video frames here.
    # Use text-only Qwen generation through the same model.

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
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
            max_new_tokens=MAX_NEW_TOKENS_SUMMARY,
            do_sample=False
        )

    generated_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(
            inputs["input_ids"],
            generated_ids
        )
    ]

    output = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    report = parse_json_response(
        output
    )

    if not isinstance(report, dict):

        report = {
            "incident_summary":
                f"{classification} detected.",

            "threat_level":
                "MEDIUM",

            "recommended_action":
                "Review the detected incident."
        }

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video",
        type=str
    )

    args = parser.parse_args()

    video_path = Path(
        args.video
    )

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    print("=" * 70)
    print(
        "SENTINELAI - QWEN2.5-VL MVP"
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
    # VIDEO
    # ========================================================

    fps, total_frames, duration = (
        get_video_info(
            video_path
        )
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
        WINDOW_SECONDS
        - OVERLAP_SECONDS
    )

    windows = []

    start = 0.0

    while start < duration:

        end = min(
            start + WINDOW_SECONDS,
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
        "Windows:",
        len(windows)
    )

    print(
        "Window:",
        WINDOW_SECONDS,
        "seconds"
    )

    print(
        "Overlap:",
        OVERLAP_SECONDS,
        "seconds"
    )

    # ========================================================
    # LOAD QWEN
    # ========================================================

    print()
    print(
        "Loading Qwen2.5-VL..."
    )

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

    print(
        "✅ Qwen loaded."
    )

    # ========================================================
    # SCAN
    # ========================================================

    all_windows = []

    pipeline_start = time.time()

    for index, (
        start,
        end
    ) in enumerate(
        windows,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(windows)}] "
            f"{start:.1f}s → {end:.1f}s"
        )

        window_start = time.time()

        frames = sample_window(
            video_path,
            start,
            end
        )

        raw_output = qwen_generate(
            model,
            processor,
            frames,
            DETECTION_PROMPT,
            MAX_NEW_TOKENS_DETECTION
        )

        detection = parse_detection(
            raw_output
        )

        classification = (
            classify_detection(
                detection
            )
        )

        elapsed = (
            time.time()
            - window_start
        )

        result = {
            "window_index":
                index,

            "start":
                start,

            "end":
                end,

            "classification":
                classification,

            "detection":
                detection,

            "raw_output":
                raw_output,

            "inference_time":
                elapsed
        }

        all_windows.append(
            result
        )

        print(
            "Classification:",
            classification
        )

        print(
            "Fire:",
            detection[
                "fire"
            ]["detected"]
        )

        print(
            "Fight:",
            detection[
                "fight"
            ]["detected"]
        )

        print(
            "Road Accident:",
            detection[
                "road_accident"
            ]["detected"]
        )

        print(
            "Time:",
            f"{elapsed:.2f}s"
        )

    # ========================================================
    # EPISODES
    # ========================================================

    incident_windows = [
        x
        for x in all_windows
        if x["classification"]
        in {
            "Fire",
            "Fight",
            "Road Accident",
            "Multiple"
        }
    ]

    episodes = build_episodes(
        incident_windows
    )

    episode_results = []

    for episode in episodes:

        episode_results.append(
            classify_episode(
                episode
            )
        )

    # ========================================================
    # FINAL CLASSIFICATION
    # ========================================================

    if episode_results:

        primary_episode = (
            episode_results[0]
        )

        final_classification = (
            primary_episode[
                "primary_classification"
            ]
        )

    else:

        primary_episode = None

        final_classification = (
            "Normal"
        )

    # ========================================================
    # INCIDENT REPORT
    # ========================================================

    if primary_episode:

        print()
        print(
            "Generating incident report..."
        )

        report = (
            generate_incident_report(
                model,
                processor,
                primary_episode
            )
        )

    else:

        report = {
            "incident_summary":
                "No fire, fight, or road accident was detected.",

            "threat_level":
                "LOW",

            "recommended_action":
                "No immediate action required."
        }

    # ========================================================
    # FINAL RESULT
    # ========================================================

    total_time = (
        time.time()
        - pipeline_start
    )

    result = {

        "project":
            "SentinelAI",

        "model":
            MODEL_NAME,

        "video":
            str(video_path),

        "duration_seconds":
            duration,

        "fps":
            fps,

        "windows_analyzed":
            len(all_windows),

        "window_seconds":
            WINDOW_SECONDS,

        "overlap_seconds":
            OVERLAP_SECONDS,

        "classification":
            final_classification,

        "incident_start_seconds":
            (
                primary_episode[
                    "start_seconds"
                ]
                if primary_episode
                else None
            ),

        "incident_end_seconds":
            (
                primary_episode[
                    "end_seconds"
                ]
                if primary_episode
                else None
            ),

        "observations":
            (
                primary_episode[
                    "observations"
                ]
                if primary_episode
                else []
            ),

        "evidence":
            (
                primary_episode[
                    "evidence"
                ]
                if primary_episode
                else []
            ),

        "incident_summary":
            report.get(
                "incident_summary",
                ""
            ),

        "threat_level":
            report.get(
                "threat_level",
                "LOW"
            ),

        "recommended_action":
            report.get(
                "recommended_action",
                ""
            ),

        "total_processing_time_seconds":
            total_time,

        "episodes":
            episode_results,

        "windows":
            all_windows
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
        /
        (
            video_path.stem
            +
            "_sentinelai_result.json"
        )
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "SENTINELAI FINAL RESULT"
    )
    print("=" * 70)

    print()

    print(
        "Classification:",
        final_classification
    )

    if primary_episode:

        print(
            "Incident:",
            f"{primary_episode['start_seconds']:.1f}s "
            f"→ "
            f"{primary_episode['end_seconds']:.1f}s"
        )

    print(
        "Threat:",
        report.get(
            "threat_level",
            "LOW"
        )
    )

    print(
        "Summary:",
        report.get(
            "incident_summary",
            ""
        )
    )

    print(
        "Action:",
        report.get(
            "recommended_action",
            ""
        )
    )

    print()
    print(
        "Processing time:",
        f"{total_time:.2f}s"
    )

    print()
    print(
        "Result saved:",
        output_path
    )

    print()
    print("=" * 70)
    print(
        "✅ SENTINELAI MVP PIPELINE COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()