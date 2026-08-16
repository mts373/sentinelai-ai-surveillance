import argparse
import gc
import json
import re
from pathlib import Path

import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# SENTINELAI - MEMORY CONTROLLED QWEN2.5-VL ENGINE
# ============================================================

MODEL_NAME = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TEMPORAL CONFIGURATION
# ============================================================

WINDOW_SECONDS = 10.0

OVERLAP_SECONDS = 5.0

# Approximately 8 frames for a 10-second window.
#
# 10 seconds × 0.8 FPS ≈ 8 frames
#
# This is intentionally lower than our previous 16-frame
# implementation to reduce visual-token and attention memory.
VIDEO_FPS = 0.8

NUM_FRAMES_APPROX = 8


# ============================================================
# VISUAL TOKEN CONTROL
# ============================================================

# Qwen visual tokens operate around 28x28 patches.
#
# We deliberately use a conservative maximum for the
# first stable experiment on the RTX 4090.
#
# 768 visual tokens maximum per sampled frame.
#
# IMPORTANT:
# This is NOT the video's original resolution.
# Qwen will control the visual representation.
MIN_PIXELS = 256 * 28 * 28

MAX_PIXELS = 768 * 28 * 28


# ============================================================
# GENERATION CONFIGURATION
# ============================================================

MAX_NEW_TOKENS = 160


# ============================================================
# SENTINELAI CLASSES
# ============================================================

CLASSES = [
    "Normal",
    "Fire",
    "Fight",
    "Road Accident",
]


# ============================================================
# DETECTION PROMPT
# ============================================================

DETECTION_PROMPT = """
You are SentinelAI, an AI CCTV incident detection system.

Analyze the provided CCTV video window.

Classify the video into EXACTLY ONE of these four classes:

1. Normal
2. Fire
3. Fight
4. Road Accident

DEFINITIONS

Normal:
No visible fire, physical fight, or road accident.

Fire:
Visible flames, burning objects, or an active fire.

Fight:
Two or more people are physically fighting,
attacking, hitting, kicking, punching, or clearly
engaged in a physical altercation.

Road Accident:
A vehicle collision, vehicle striking a person,
vehicle striking another vehicle or object, or a
clearly visible traffic accident.

IMPORTANT RULES

Do NOT classify people as fighting merely because:
- they are standing close together
- they are walking together
- they are running
- they are talking
- they are gesturing
- there is a crowd

Do NOT classify Road Accident merely because:
- vehicles are moving
- vehicles are close together
- traffic is heavy
- a vehicle is driving toward the camera

Do NOT classify Fire because of:
- red/orange objects
- lights
- clothing
- sunset
- reflections

Only report an incident when there is visible evidence.

Return ONLY valid JSON.

Required format:

{
    "classification": "Normal | Fire | Fight | Road Accident",
    "evidence": "short description of visible evidence",
    "incident_summary": "short description of what happened"
}
"""


# ============================================================
# GPU MEMORY
# ============================================================

def print_gpu_memory(
    prefix=""
):

    if not torch.cuda.is_available():
        return

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print(
        f"{prefix}"
        f"GPU allocated: {allocated:.2f} GB | "
        f"reserved: {reserved:.2f} GB | "
        f"peak: {peak:.2f} GB"
    )


def clear_gpu_memory():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print(
        "SENTINELAI - MEMORY CONTROLLED "
        "QWEN2.5-VL"
    )
    print("=" * 70)

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU memory:",
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    print()
    print(
        "Model:",
        MODEL_NAME
    )

    print(
        "Video sampling FPS:",
        VIDEO_FPS
    )

    print(
        "Approx frames / 10s:",
        NUM_FRAMES_APPROX
    )

    print(
        "Maximum visual pixels:",
        MAX_PIXELS
    )

    print()
    print(
        "Loading Qwen2.5-VL..."
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    if DEVICE == "cuda":

        model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                MODEL_NAME,

                torch_dtype=torch.float16,

                device_map="auto",

                low_cpu_mem_usage=True,
            )
        )

    else:

        model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                MODEL_NAME,

                torch_dtype=torch.float32,

                low_cpu_mem_usage=True,
            )
        )

    model.eval()

    # --------------------------------------------------------
    # PROCESSOR
    # --------------------------------------------------------

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_NAME,

            min_pixels=MIN_PIXELS,

            max_pixels=MAX_PIXELS,
        )
    )

    print()
    print(
        "Model loaded."
    )

    print(
        "Processor loaded."
    )

    print_gpu_memory(
        "After model loading → "
    )

    return model, processor


# ============================================================
# BUILD VIDEO MESSAGE
# ============================================================

def build_video_message(
    video_path
):

    video_path = Path(
        video_path
    ).resolve()

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video does not exist:\n"
            f"{video_path}"
        )

    # IMPORTANT:
    #
    # Use the normal Windows path.
    #
    # DO NOT use:
    #
    # video_path.as_uri()
    #
    # because decord on Windows can fail with encoded
    # file:/// URLs for paths containing spaces, &, etc.

    messages = [

        {
            "role": "user",

            "content": [

                {
                    "type": "video",

                    "video":
                        str(video_path),

                    # Approximately 8 frames for a
                    # 10-second window.

                    "fps":
                        float(VIDEO_FPS),

                    # Prevent high-resolution videos
                    # from generating an uncontrolled
                    # number of visual tokens.

                    "max_pixels":
                        MAX_PIXELS,
                },

                {
                    "type": "text",

                    "text":
                        DETECTION_PROMPT,
                },
            ],
        }
    ]

    return messages


# ============================================================
# QWEN VIDEO INFERENCE
# ============================================================

@torch.inference_mode()
def qwen_generate_video(
    model,
    processor,
    video_path,
):

    # --------------------------------------------------------
    # CLEAN MEMORY BEFORE EVERY WINDOW
    # --------------------------------------------------------

    clear_gpu_memory()

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    print()
    print_gpu_memory(
        "Before inference → "
    )

    # --------------------------------------------------------
    # MESSAGE
    # --------------------------------------------------------

    messages = build_video_message(
        video_path
    )

    # --------------------------------------------------------
    # CHAT TEMPLATE
    # --------------------------------------------------------

    text = processor.apply_chat_template(
        messages,

        tokenize=False,

        add_generation_prompt=True,
    )

    # --------------------------------------------------------
    # QWEN VIDEO PROCESSING
    #
    # IMPORTANT:
    #
    # We intentionally DO NOT use:
    #
    # return_video_kwargs=True
    #
    # because your installed qwen-vl-utils /
    # Transformers combination returns fps as:
    #
    # [0.768]
    #
    # while Transformers 5.15 expects:
    #
    # 0.768
    #
    # --------------------------------------------------------

    vision_result = process_vision_info(
        messages
    )

    # qwen-vl-utils normally returns:
    #
    # image_inputs, video_inputs
    #
    image_inputs = vision_result[0]

    video_inputs = vision_result[1]

    # --------------------------------------------------------
    # PROCESS INPUT
    # --------------------------------------------------------

    inputs = processor(

        text=[text],

        images=image_inputs,

        videos=video_inputs,

        padding=True,

        return_tensors="pt",
    )

    # --------------------------------------------------------
    # MOVE TO GPU
    # --------------------------------------------------------

    if DEVICE == "cuda":

        inputs = inputs.to(
            "cuda"
        )

    else:

        inputs = inputs.to(
            "cpu"
        )

    print_gpu_memory(
        "Inputs prepared → "
    )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    generated_ids = model.generate(

        **inputs,

        max_new_tokens=
            MAX_NEW_TOKENS,

        do_sample=False,
    )

    # --------------------------------------------------------
    # REMOVE INPUT TOKENS
    # --------------------------------------------------------

    generated_ids_trimmed = [

        output_ids[
            len(input_ids):
        ]

        for input_ids, output_ids
        in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    output_text = (
        processor.batch_decode(

            generated_ids_trimmed,

            skip_special_tokens=True,

            clean_up_tokenization_spaces=False,
        )
    )

    result = output_text[0]

    # --------------------------------------------------------
    # DELETE LARGE TEMPORARY OBJECTS
    # --------------------------------------------------------

    del inputs

    del generated_ids

    del generated_ids_trimmed

    del image_inputs

    del video_inputs

    del vision_result

    del messages

    clear_gpu_memory()

    print_gpu_memory(
        "After cleanup → "
    )

    return result


# ============================================================
# PARSE QWEN JSON
# ============================================================

def parse_json_output(
    text
):

    text = text.strip()

    # --------------------------------------------------------
    # DIRECT JSON
    # --------------------------------------------------------

    try:

        data = json.loads(
            text
        )

        return normalize_result(
            data
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # JSON INSIDE MARKDOWN
    # --------------------------------------------------------

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if match:

        try:

            data = json.loads(
                match.group(0)
            )

            return normalize_result(
                data
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return {

        "classification":
            "Normal",

        "evidence":
            text[:500],

        "incident_summary":
            text[:500],
    }


# ============================================================
# NORMALIZE CLASSIFICATION
# ============================================================

def normalize_result(
    data
):

    classification = str(
        data.get(
            "classification",
            "Normal"
        )
    ).strip()

    lower = (
        classification
        .lower()
    )

    # --------------------------------------------------------
    # ROAD ACCIDENT
    # --------------------------------------------------------

    if (
        "road" in lower
        and (
            "accident" in lower
            or "collision" in lower
            or "crash" in lower
        )
    ):

        classification = (
            "Road Accident"
        )

    # --------------------------------------------------------
    # FIRE
    # --------------------------------------------------------

    elif "fire" in lower:

        classification = "Fire"

    # --------------------------------------------------------
    # FIGHT
    # --------------------------------------------------------

    elif "fight" in lower:

        classification = "Fight"

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    elif "normal" in lower:

        classification = "Normal"

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    else:

        classification = "Normal"

    return {

        "classification":
            classification,

        "evidence":
            str(
                data.get(
                    "evidence",
                    ""
                )
            ),

        "incident_summary":
            str(
                data.get(
                    "incident_summary",
                    ""
                )
            ),
    }


# ============================================================
# SINGLE VIDEO TEST
# ============================================================

def test_video(
    video_path
):

    model = None

    processor = None

    try:

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        model, processor = (
            load_model()
        )

        print()
        print("=" * 70)
        print(
            "RUNNING MEMORY-SAFE VIDEO TEST"
        )
        print("=" * 70)

        print(
            "Video:",
            video_path
        )

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        raw_output = (
            qwen_generate_video(

                model=model,

                processor=processor,

                video_path=video_path,
            )
        )

        # ----------------------------------------------------
        # RAW OUTPUT
        # ---------------------------------------------------

        print()
        print("=" * 70)
        print(
            "RAW QWEN OUTPUT"
        )
        print("=" * 70)

        print(
            raw_output
        )

        # ----------------------------------------------------
        # PARSED RESULT
        # ----------------------------------------------------

        result = (
            parse_json_output(
                raw_output
            )
        )

        print()
        print("=" * 70)
        print(
            "PARSED SENTINELAI RESULT"
        )
        print("=" * 70)

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

        return result

    finally:

        # ----------------------------------------------------
        # FINAL MODEL CLEANUP
        # ----------------------------------------------------

        if model is not None:

            del model

        if processor is not None:

            del processor

        clear_gpu_memory()

        print()
        print_gpu_memory(
            "After model cleanup → "
        )

        print()
        print("=" * 70)
        print(
            "TEST FINISHED"
        )
        print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "SentinelAI memory-controlled "
            "Qwen2.5-VL video inference"
        )
    )

    parser.add_argument(
        "video",
        type=str,
        help="Full path to video file",
    )

    args = parser.parse_args()

    video_path = Path(
        args.video
    ).resolve()

    if not video_path.exists():

        raise FileNotFoundError(
            f"\nVideo not found:\n"
            f"{video_path}\n"
        )

    test_video(
        video_path
    )


if __name__ == "__main__":

    main()