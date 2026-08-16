import os
import json
import gc
from pathlib import Path
from collections import Counter
from copy import deepcopy

# Reduce CUDA allocator fragmentation before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# SENTINELAI - CORRECTIVE QLoRA SFT v3
#
# Designed for RTX 4090 24 GB.
#
# IMPORTANT DIFFERENCE FROM THE FAILED RUN:
#   Every corrective example MUST describe its exact temporal
#   window (start/end). The collator sends only that window to
#   Qwen, instead of the whole source video.
#
# Memory controls:
#   - 0.5 FPS target for 10 s windows
#   - max 8 frames
#   - max 401408 pixels/frame
#   - total video pixels capped
#   - batch=1
#   - gradient accumulation=8
#   - gradient checkpointing
#   - 4-bit NF4 QLoRA
#
# Output is NEW and never overwrites the baseline adapter.
# ============================================================

PROJECT_ROOT = Path(r"C:\SentinelAI_Qwen")

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

DATASET_PATH = (
    PROJECT_ROOT / "dataset" / "sft" / "corrective_sft_train.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "models" / "sentinelai_qwen25vl_lora_corrective_v3"
)

# -------------------------
# Training
# -------------------------
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 5e-5
WARMUP_STEPS = 2

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# -------------------------
# Video memory controls
# -------------------------
VIDEO_FPS = 0.5
MAX_VIDEO_FRAMES = 8

# 128*28^2 and 512*28^2.
MIN_PIXELS = 100352
MAX_PIXELS = 401408

# Qwen recommends controlling total video tokens/pixels.
# 16384*28*28 is a conservative cap for this 24 GB card.
TOTAL_PIXELS = 16384 * 28 * 28


# ============================================================
# HELPERS
# ============================================================

def gpu_memory(label):
    if not torch.cuda.is_available():
        return
    print(
        f"{label} → "
        f"allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB | "
        f"reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB | "
        f"peak: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB"
    )


def load_data():
    print("\nLoading corrective SFT dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("corrective_sft_train.json must be a JSON list.")

    print(f"Records: {len(data)}")

    counts = Counter()
    for i, item in enumerate(data):
        if "messages" not in item:
            raise ValueError(f"Record {i}: missing messages.")
        if "label" not in item:
            raise ValueError(f"Record {i}: missing label.")

        # We accept either:
        #   video + start/end
        # or a video object inside messages containing start/end.
        video_path, start, end = get_window_metadata(item)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(
                f"Record {i}: video does not exist:\n{video_path}"
            )

        if start is None or end is None:
            raise ValueError(
                f"Record {i}: exact start/end window is missing.\n"
                f"Video: {video_path}\n"
                "Do NOT train this corrective dataset until the "
                "31 human corrections contain their exact windows."
            )

        if end <= start:
            raise ValueError(
                f"Record {i}: invalid window {start} -> {end}"
            )

        counts[item["label"]] += 1

    print("Distribution:")
    for label in ["Normal", "Fire", "Fight", "Road Accident"]:
        print(f"  {label:<15}{counts[label]}")

    print("All corrective records have valid exact windows.")
    return data


def get_window_metadata(item):
    """
    Find video path + exact temporal window without changing the
    original JSON structure.
    """

    video_path = item.get("video_path") or item.get("video")
    start = item.get("start")
    end = item.get("end")

    # Sometimes the metadata is attached to the video content item.
    if video_path is None or start is None or end is None:
        for msg in item.get("messages", []):
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue

            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "video":
                    continue

                if video_path is None:
                    video_path = part.get("video")

                if start is None:
                    start = part.get("video_start", part.get("start"))

                if end is None:
                    end = part.get("video_end", part.get("end"))

    # Normalize file:// paths.
    if isinstance(video_path, str) and video_path.startswith("file:///"):
        video_path = video_path[8:].replace("/", "\\")

    return video_path, start, end


def build_window_messages(item):
    """
    Copy the original messages and force every video item to refer
    to the exact human-labelled temporal window.

    Also applies conservative video sampling/resolution limits.
    """

    messages = deepcopy(item["messages"])

    video_path, start, end = get_window_metadata(item)

    found = False

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "video" and "video" not in part:
                continue

            found = True

            # Use the actual source file from the dataset metadata.
            part["video"] = video_path

            # Exact human-labelled temporal window.
            part["video_start"] = float(start)
            part["video_end"] = float(end)

            # Temporal sampling.
            part["fps"] = VIDEO_FPS

            # Spatial/total visual-token controls.
            part["min_pixels"] = MIN_PIXELS
            part["max_pixels"] = MAX_PIXELS
            part["total_pixels"] = TOTAL_PIXELS

    if not found:
        raise ValueError(
            f"No video content found in messages for {video_path}"
        )

    return messages


# ============================================================
# COLLATOR
# ============================================================

class CorrectiveVideoCollator:

    def __init__(self, processor):
        self.processor = processor
        self.debug_printed = False

    def __call__(self, examples):

        if len(examples) != 1:
            raise RuntimeError("Batch size must remain exactly 1.")

        item = examples[0]
        messages = build_window_messages(item)

        # -------------------------
        # Full conversation
        # -------------------------
        full_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        if video_inputs is None:
            raise RuntimeError(
                f"Video decoding returned None:\n{item}"
            )

        inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # -------------------------
        # User-only prompt
        # -------------------------
        user_messages = [messages[0]]

        user_text = self.processor.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        user_image_inputs, user_video_inputs = process_vision_info(
            user_messages
        )

        prompt_inputs = self.processor(
            text=[user_text],
            images=user_image_inputs,
            videos=user_video_inputs,
            padding=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_ids = prompt_inputs["input_ids"]

        if prompt_ids.shape[-1] >= input_ids.shape[-1]:
            raise RuntimeError(
                f"Prompt is not shorter than full input: "
                f"{prompt_ids.shape[-1]} >= {input_ids.shape[-1]}"
            )

        labels = input_ids.clone()
        prompt_length = prompt_ids.shape[-1]
        labels[:, :prompt_length] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        trainable_tokens = (labels != -100).sum().item()
        if trainable_tokens <= 0:
            raise RuntimeError("ZERO TRAINABLE TOKENS.")

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        for key in [
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
        ]:
            if key in inputs:
                batch[key] = inputs[key]

        if not self.debug_printed:
            self.debug_printed = True

            video_path, start, end = get_window_metadata(item)

            print("\n" + "=" * 70)
            print("FIRST CORRECTIVE TRAINING BATCH")
            print("=" * 70)
            print(f"Video: {video_path}")
            print(f"Human window: {float(start):.2f}s -> {float(end):.2f}s")
            print(f"Label: {item['label']}")
            print(f"Input tokens: {input_ids.shape[-1]}")
            print(f"Prompt tokens masked: {prompt_length}")
            print(f"Assistant tokens for loss: {trainable_tokens}")

            if "pixel_values_videos" in inputs:
                print(
                    "pixel_values_videos: "
                    f"{tuple(inputs['pixel_values_videos'].shape)}"
                )

            if "video_grid_thw" in inputs:
                print(
                    f"video_grid_thw: {inputs['video_grid_thw']}"
                )

            print(f"Target FPS: {VIDEO_FPS}")
            print(f"Max video frames: {MAX_VIDEO_FRAMES}")
            print(f"Max pixels/frame: {MAX_PIXELS}")
            print(f"Total pixels cap: {TOTAL_PIXELS}")
            print("=" * 70)

        return batch


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("SENTINELAI - MEMORY-OPTIMIZED CORRECTIVE QLoRA")
    print("=" * 70)

    print(f"Dataset: {DATASET_PATH}")
    print(f"Output:  {OUTPUT_DIR}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB"
    )

    torch.cuda.reset_peak_memory_stats()

    data = load_data()

    # -------------------------
    # Quantization
    # -------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("\nLoading Qwen2.5-VL...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    # Do NOT enable flash_attention_2 blindly:
    # it requires the separate flash-attn package and can break an
    # otherwise working environment. SDPA is already the working
    # attention implementation in the user's traceback.

    gpu_memory("After model loading")

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("Preparing QLoRA...")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = CorrectiveVideoCollator(processor)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=1,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        learning_rate=LEARNING_RATE,

        warmup_steps=WARMUP_STEPS,

        logging_steps=1,
        logging_first_step=True,

        save_strategy="epoch",
        save_total_limit=1,

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        optim="paged_adamw_8bit",

        weight_decay=0.01,

        lr_scheduler_type="cosine",

        remove_unused_columns=False,

        dataloader_num_workers=0,

        report_to="none",

        seed=42,

        # Avoid unnecessary Trainer memory overhead.
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
        data_collator=collator,
    )

    print("\n" + "=" * 70)
    print("STARTING MEMORY-OPTIMIZED CORRECTIVE TRAINING")
    print("=" * 70)
    print(f"Records: {len(data)}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Effective batch: {GRADIENT_ACCUMULATION}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"FPS: {VIDEO_FPS}")
    print(f"Max frames: {MAX_VIDEO_FRAMES}")
    print(f"Max pixels: {MAX_PIXELS}")
    print(f"Total pixels: {TOTAL_PIXELS}")
    gpu_memory("Before training")

    try:
        result = trainer.train()

    except torch.OutOfMemoryError:
        print("\n" + "=" * 70)
        print("CUDA OOM")
        print("=" * 70)
        gpu_memory("OOM state")
        print(
            "\nThis run was intentionally not allowed to continue."
        )
        raise

    metrics = result.metrics

    train_loss = metrics.get("train_loss")
    if train_loss is None or train_loss == 0:
        raise RuntimeError(
            f"Invalid train_loss: {train_loss}"
        )

    print("\n" + "=" * 70)
    print("TRAINING METRICS")
    print("=" * 70)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nSaving corrective adapter...")

    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    metrics_file = OUTPUT_DIR / "training_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    gpu_memory("After training")

    del trainer
    del model
    del processor

    gc.collect()
    torch.cuda.empty_cache()

    gpu_memory("After cleanup")

    print("\n" + "=" * 70)
    print("CORRECTIVE TRAINING FINISHED SUCCESSFULLY")
    print("=" * 70)
    print(f"Adapter: {OUTPUT_DIR}")
    print(f"Metrics: {metrics_file}")


if __name__ == "__main__":
    main()