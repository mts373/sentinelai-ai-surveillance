from pathlib import Path
import json
import time

import cv2
import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

VIDEO_PATH = Path(
    r"C:\UCF_CRIME_ORIGINAL\Anomaly-Videos-Part-1\Arson\Arson040_x264.mp4"
)

WINDOW_START_SECONDS = 0
WINDOW_DURATION_SECONDS = 10

NUM_FRAMES = 16


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("=" * 70)
print("SENTINELAI - QWEN TEMPORAL WINDOW TEST")
print("=" * 70)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECK VIDEO
# ============================================================

print()
print("=" * 70)
print("VIDEO")
print("=" * 70)

print()
print("Video:", VIDEO_PATH)

if not VIDEO_PATH.exists():

    raise FileNotFoundError(
        f"Video not found:\n{VIDEO_PATH}"
    )

print("✅ Video exists.")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = float(
    cap.get(cv2.CAP_PROP_FPS)
)

duration = (
    total_frames / fps
    if fps > 0
    else 0
)


print()
print("FPS:", fps)
print("Total frames:", total_frames)
print(
    "Video duration:",
    f"{duration:.2f} seconds"
)


# ============================================================
# WINDOW INFORMATION
# ============================================================

window_start_frame = int(
    WINDOW_START_SECONDS * fps
)

window_end_seconds = min(
    WINDOW_START_SECONDS
    + WINDOW_DURATION_SECONDS,
    duration
)

window_end_frame = int(
    window_end_seconds * fps
)

window_frames = (
    window_end_frame
    - window_start_frame
)


print()
print("=" * 70)
print("TEMPORAL WINDOW")
print("=" * 70)

print(
    "Start:",
    f"{WINDOW_START_SECONDS:.2f}s"
)

print(
    "End:",
    f"{window_end_seconds:.2f}s"
)

print(
    "Window frames:",
    window_frames
)


# ============================================================
# SAMPLE FRAMES FROM WINDOW
# ============================================================

if window_frames <= 0:

    cap.release()

    raise RuntimeError(
        "Temporal window contains no frames."
    )


indices = [
    int(x)
    for x in torch.linspace(
        window_start_frame,
        window_end_frame - 1,
        NUM_FRAMES
    ).tolist()
]


frames = []


print()
print(
    f"Sampling {NUM_FRAMES} frames..."
)


for frame_index in indices:

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index
    )

    success, frame = cap.read()

    if not success:

        cap.release()

        raise RuntimeError(
            f"Failed to read frame "
            f"{frame_index}"
        )


    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frames.append(
        frame
    )


cap.release()


print(
    "Sampled frames:",
    len(frames)
)

print(
    "Frame shape:",
    frames[0].shape
)


# ============================================================
# CONVERT FRAMES TO PIL
# ============================================================

from PIL import Image


pil_frames = [
    Image.fromarray(frame)
    for frame in frames
]


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING QWEN2.5-VL")
print("=" * 70)

print()

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
print("✅ Model loaded.")


# ============================================================
# BUILD MESSAGE
# ============================================================

prompt = """
You are SentinelAI, a CCTV security analysis system.

Analyze the provided 10-second CCTV video segment.

You MUST classify the segment into exactly ONE of these four classes:

1. Normal
2. Fire
3. Fight
4. Road Accident

Do NOT use any other class.

Important:
- Base the classification only on visible evidence.
- If there is no clear fire, fight, or road accident, classify it as Normal.
- Do not assume an incident just because the scene looks unusual.

Return ONLY valid JSON using exactly this structure:

{
  "classification": "Normal",
  "evidence": "brief visual evidence",
  "incident_summary": "brief summary",
  "threat_level": "LOW"
}

Threat level must be one of:
LOW
MEDIUM
HIGH
"""


content = [
    {
        "type": "text",
        "text": prompt
    }
]


# ============================================================
# ADD FRAMES
# ============================================================

for frame in pil_frames:

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


# ============================================================
# APPLY CHAT TEMPLATE
# ============================================================

print()
print("=" * 70)
print("PREPARING QWEN INPUT")
print("=" * 70)


text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


inputs = processor(
    text=[text],
    images=pil_frames,
    padding=True,
    return_tensors="pt"
)


# ============================================================
# MOVE INPUT TO GPU
# ============================================================

inputs = {
    key: value.to(DEVICE)
    if torch.is_tensor(value)
    else value
    for key, value in inputs.items()
}


print(
    "Input prepared."
)

if "input_ids" in inputs:

    print(
        "Input token shape:",
        inputs["input_ids"].shape
    )


# ============================================================
# GENERATION
# ============================================================

print()
print("=" * 70)
print("RUNNING QWEN INFERENCE")
print("=" * 70)

print()

start_time = time.time()


with torch.inference_mode():

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False
    )


inference_time = (
    time.time()
    - start_time
)


# ============================================================
# REMOVE INPUT TOKENS
# ============================================================

generated_ids_trimmed = [
    output_ids[len(input_ids):]
    for input_ids, output_ids
    in zip(
        inputs["input_ids"],
        generated_ids
    )
]


output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]


# ============================================================
# DISPLAY RESULT
# ============================================================

print(
    "Inference time:",
    f"{inference_time:.2f} seconds"
)

print()

print("=" * 70)
print("QWEN RESULT")
print("=" * 70)

print()

print(output_text)


# ============================================================
# SAVE RESULT
# ============================================================

result = {
    "video": str(VIDEO_PATH),
    "window_start": WINDOW_START_SECONDS,
    "window_end": window_end_seconds,
    "num_frames": NUM_FRAMES,
    "inference_time_seconds": inference_time,
    "raw_output": output_text
}


RESULT_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "single_window_result.json"
)


RESULT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        result,
        file,
        indent=4,
        ensure_ascii=False
    )


print()
print(
    "Result saved:",
    RESULT_PATH
)

print()
print("=" * 70)
print("✅ SINGLE-WINDOW QWEN TEST COMPLETE")
print("=" * 70)