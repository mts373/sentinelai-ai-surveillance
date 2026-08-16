import json
import gc
import os
from pathlib import Path
from collections import Counter

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
# SENTINELAI
# QWEN2.5-VL 7B
# QLoRA SFT TRAINING
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\SentinelAI_Qwen"
)

MODEL_NAME = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

SFT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "sft"
    / "sft_train.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "sentinelai_qwen25vl_lora"
)


# ============================================================
# TRAINING CONFIG
# ============================================================

NUM_EPOCHS = 1

BATCH_SIZE = 1

GRADIENT_ACCUMULATION = 8

LEARNING_RATE = 1e-4

WARMUP_STEPS = 2

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05

MIN_PIXELS = 200704

MAX_PIXELS = 602112


# ============================================================
# GPU MEMORY REPORT
# ============================================================

def print_gpu_memory(label):

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
        f"{label} → "
        f"allocated: {allocated:.2f} GB | "
        f"reserved: {reserved:.2f} GB | "
        f"peak: {peak:.2f} GB"
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print()

    print(
        "Loading SFT dataset..."
    )

    if not SFT_DATASET.exists():

        raise FileNotFoundError(
            f"\nSFT dataset does not exist:\n"
            f"{SFT_DATASET}"
        )

    with open(
        SFT_DATASET,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(
        data,
        list,
    ):

        raise ValueError(
            "sft_train.json must "
            "contain a JSON list."
        )

    print(
        f"Loaded records: {len(data)}"
    )

    # --------------------------------------------------------
    # Validate records
    # --------------------------------------------------------

    print(
        "Validating videos..."
    )

    for i, item in enumerate(data):

        if "video" not in item:

            raise ValueError(
                f"Record {i} has no video field."
            )

        if "messages" not in item:

            raise ValueError(
                f"Record {i} has no messages field."
            )

        video_path = item[
            "video"
        ]

        if not os.path.isfile(
            video_path
        ):

            raise FileNotFoundError(
                f"\nMissing video:\n"
                f"{video_path}"
            )

    print(
        f"Validated {len(data)} videos."
    )

    return data


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

def print_distribution(data):

    counts = Counter(
        item.get(
            "label",
            "UNKNOWN"
        )
        for item in data
    )

    print()

    print(
        "Dataset distribution:"
    )

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


# ============================================================
# QWEN VIDEO COLLATOR
# ============================================================

class QwenVideoCollator:

    def __init__(
        self,
        processor,
    ):

        self.processor = processor

    # ========================================================
    # MAIN COLLATOR
    # ========================================================

    def __call__(
        self,
        examples,
    ):

        # ----------------------------------------------------
        # We intentionally train with batch size 1.
        # ----------------------------------------------------

        if len(examples) != 1:

            raise RuntimeError(
                "QwenVideoCollator expects "
                "exactly one example."
            )

        example = examples[0]

        messages = example[
            "messages"
        ]

        # ====================================================
        # FULL CONVERSATION
        # ====================================================

        full_text = (
            self.processor.apply_chat_template(
                messages,

                tokenize=False,

                add_generation_prompt=False,
            )
        )

        # ====================================================
        # DECODE VIDEO
        #
        # CRITICAL:
        #
        # DO NOT USE:
        #
        # return_video_kwargs=True
        #
        # DO NOT PASS:
        #
        # **video_kwargs
        #
        # Your environment returns:
        #
        #     fps = [2.0]
        #
        # Transformers 5.15 rejects that.
        #
        # The working path is:
        #
        #     process_vision_info(messages)
        # ====================================================

        (
            image_inputs,
            video_inputs,
        ) = process_vision_info(
            messages
        )

        if video_inputs is None:

            raise RuntimeError(
                f"\nVideo decoding returned None.\n"
                f"Video: {example['video']}"
            )

        # ====================================================
        # PROCESS FULL MULTIMODAL INPUT
        # ====================================================

        inputs = self.processor(

            text=[
                full_text
            ],

            images=image_inputs,

            videos=video_inputs,

            padding=True,

            return_tensors="pt",

        )

        # ====================================================
        # BUILD USER-ONLY PROMPT
        #
        # IMPORTANT:
        #
        # We include the VIDEO here too.
        #
        # This makes prompt_length correspond to the
        # actual multimodal input sequence.
        #
        # This is better than masking only the text tokens.
        # ====================================================

        user_messages = [
            messages[0]
        ]

        user_text = (
            self.processor.apply_chat_template(
                user_messages,

                tokenize=False,

                add_generation_prompt=True,
            )
        )

        # ----------------------------------------------------
        # Decode the same video for the prompt portion.
        # ----------------------------------------------------

        (
            user_image_inputs,
            user_video_inputs,
        ) = process_vision_info(
            user_messages
        )

        # ====================================================
        # PROCESS USER-ONLY MULTIMODAL PROMPT
        # ====================================================

        prompt_inputs = self.processor(

            text=[
                user_text
            ],

            images=user_image_inputs,

            videos=user_video_inputs,

            padding=True,

            return_tensors="pt",

        )

        # ====================================================
        # INPUT IDS
        # ====================================================

        input_ids = inputs[
            "input_ids"
        ]

        attention_mask = inputs[
            "attention_mask"
        ]

        prompt_input_ids = (
            prompt_inputs[
                "input_ids"
            ]
        )

        # ====================================================
        # SAFETY CHECK
        # ====================================================

        if (
            prompt_input_ids.shape[-1]
            >= input_ids.shape[-1]
        ):

            raise RuntimeError(
                "\nPrompt is not shorter "
                "than full conversation.\n"
                f"Prompt tokens: "
                f"{prompt_input_ids.shape[-1]}\n"
                f"Full tokens: "
                f"{input_ids.shape[-1]}"
            )

        # ====================================================
        # CREATE LABELS
        # ====================================================

        labels = input_ids.clone()

        prompt_length = (
            prompt_input_ids.shape[-1]
        )

        # ----------------------------------------------------
        # Mask system + user + video tokens.
        #
        # Only assistant response contributes to loss.
        # ----------------------------------------------------

        labels[
            :,
            :prompt_length
        ] = -100

        # ====================================================
        # MASK PADDING
        # ====================================================

        pad_token_id = (
            self.processor
            .tokenizer
            .pad_token_id
        )

        if pad_token_id is not None:

            labels[
                labels == pad_token_id
            ] = -100

        # ====================================================
        # TRAINABLE TOKEN CHECK
        # ====================================================

        trainable_tokens = (
            labels != -100
        ).sum().item()

        if trainable_tokens <= 0:

            raise RuntimeError(
                "\nZERO TRAINABLE TOKENS.\n"
                "The assistant response is not "
                "being included in the loss."
            )

        # ====================================================
        # BUILD BATCH
        # ====================================================

        batch = {

            "input_ids":
                input_ids,

            "attention_mask":
                attention_mask,

            "labels":
                labels,

        }

        # ====================================================
        # ADD VIDEO TENSORS
        # ====================================================

        multimodal_keys = [

            "pixel_values_videos",

            "video_grid_thw",

            "second_per_grid_ts",

            "pixel_values",

            "image_grid_thw",

            "mm_token_type_ids",

        ]

        for key in multimodal_keys:

            if key in inputs:

                batch[key] = inputs[
                    key
                ]

        # ====================================================
        # FIRST-BATCH DEBUG INFORMATION
        # ====================================================

        if not hasattr(
            self,
            "_printed_debug",
        ):

            self._printed_debug = True

            print()

            print(
                "=" * 70
            )

            print(
                "FIRST TRAINING BATCH"
            )

            print(
                "=" * 70
            )

            print(
                f"Video: "
                f"{example['video']}"
            )

            print(
                f"Label: "
                f"{example.get('label')}"
            )

            print(
                f"Input tokens: "
                f"{input_ids.shape[-1]}"
            )

            print(
                f"Prompt tokens masked: "
                f"{prompt_length}"
            )

            print(
                f"Assistant tokens for loss: "
                f"{trainable_tokens}"
            )

            if (
                "pixel_values_videos"
                in inputs
            ):

                print(
                    "pixel_values_videos: "
                    f"{tuple(inputs['pixel_values_videos'].shape)}"
                )

            if (
                "video_grid_thw"
                in inputs
            ):

                print(
                    "video_grid_thw: "
                    f"{inputs['video_grid_thw']}"
                )

            print()

            print(
                "FPS kwargs: NOT USED"
            )

            print(
                "First batch preprocessing: OK"
            )

            print(
                "=" * 70
            )

        return batch


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "SENTINELAI - QWEN2.5-VL "
        "QLORA SFT"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"SFT dataset: "
        f"{SFT_DATASET}"
    )

    print(
        f"Output: "
        f"{OUTPUT_DIR}"
    )

    # ========================================================
    # GPU
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "GPU"
    )

    print(
        "=" * 70
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        "CUDA: True"
    )

    print(
        "GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    total_memory = (
        torch.cuda
        .get_device_properties(0)
        .total_memory
        / (1024 ** 3)
    )

    print(
        f"GPU memory: "
        f"{total_memory:.2f} GB"
    )

    torch.cuda.reset_peak_memory_stats()

    # ========================================================
    # DATA
    # ========================================================

    data = load_dataset()

    print_distribution(
        data
    )

    # ========================================================
    # 4-BIT QLORA
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "4-BIT QLoRA"
    )

    print(
        "=" * 70
    )

    bnb_config = (
        BitsAndBytesConfig(

            load_in_4bit=True,

            bnb_4bit_quant_type="nf4",

            bnb_4bit_use_double_quant=True,

            bnb_4bit_compute_dtype=(
                torch.bfloat16
            ),

        )
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()

    print(
        "Loading Qwen2.5-VL..."
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(

            MODEL_NAME,

            quantization_config=(
                bnb_config
            ),

            device_map={
                "": 0
            },

            torch_dtype=(
                torch.bfloat16
            ),

            low_cpu_mem_usage=True,

        )
    )

    model.config.use_cache = False

    print_gpu_memory(
        "After model loading"
    )

    # ========================================================
    # PROCESSOR
    # ========================================================

    print()

    print(
        "Loading processor..."
    )

    processor = (
        AutoProcessor.from_pretrained(

            MODEL_NAME,

            min_pixels=MIN_PIXELS,

            max_pixels=MAX_PIXELS,

        )
    )

    # ========================================================
    # PAD TOKEN
    # ========================================================

    if (
        processor.tokenizer.pad_token
        is None
    ):

        processor.tokenizer.pad_token = (
            processor.tokenizer.eos_token
        )

    # ========================================================
    # PREPARE QLORA
    # ========================================================

    print()

    print(
        "Preparing model for QLoRA..."
    )

    model = (
        prepare_model_for_kbit_training(
            model
        )
    )

    # ========================================================
    # LORA
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "LORA CONFIGURATION"
    )

    print(
        "=" * 70
    )

    print(
        f"LoRA rank: "
        f"{LORA_R}"
    )

    print(
        f"LoRA alpha: "
        f"{LORA_ALPHA}"
    )

    print(
        f"LoRA dropout: "
        f"{LORA_DROPOUT}"
    )

    lora_config = (
        LoraConfig(

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
    )

    model = get_peft_model(
        model,
        lora_config
    )

    model.print_trainable_parameters()

    # ========================================================
    # COLLATOR
    # ========================================================

    collator = (
        QwenVideoCollator(
            processor
        )
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # TRAINING ARGUMENTS
    # ========================================================

    training_args = (
        TrainingArguments(

            output_dir=str(
                OUTPUT_DIR
            ),

            num_train_epochs=(
                NUM_EPOCHS
            ),

            per_device_train_batch_size=(
                BATCH_SIZE
            ),

            gradient_accumulation_steps=(
                GRADIENT_ACCUMULATION
            ),

            learning_rate=(
                LEARNING_RATE
            ),

            warmup_steps=(
                WARMUP_STEPS
            ),

            logging_steps=1,

            logging_first_step=True,

            save_strategy="epoch",

            save_total_limit=2,

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

        )
    )

    # ========================================================
    # TRAINER
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "CREATING TRANSFORMERS TRAINER"
    )

    print(
        "=" * 70
    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=data,

        data_collator=collator,

    )

    # ========================================================
    # TRAINING CONFIGURATION
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "STARTING QLORA TRAINING"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Epochs: "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION}"
    )

    print(
        f"Effective batch size: "
        f"{BATCH_SIZE * GRADIENT_ACCUMULATION}"
    )

    print(
        "Video decoder: decord"
    )

    print(
        "Video kwargs: DISABLED"
    )

    print(
        f"Min pixels: "
        f"{MIN_PIXELS}"
    )

    print(
        f"Max pixels: "
        f"{MAX_PIXELS}"
    )

    print()

    print_gpu_memory(
        "Before training"
    )

    # ========================================================
    # TRAIN
    # ========================================================

    try:

        result = trainer.train()

    except Exception:

        print()

        print(
            "=" * 70
        )

        print(
            "TRAINING FAILED"
        )

        print(
            "=" * 70
        )

        print_gpu_memory(
            "Failure memory state"
        )

        raise

    # ========================================================
    # METRICS
    # ========================================================

    metrics = result.metrics

    print()

    print(
        "=" * 70
    )

    print(
        "TRAINING METRICS"
    )

    print(
        "=" * 70
    )

    for key, value in metrics.items():

        print(
            f"{key}: {value}"
        )

    train_loss = metrics.get(
        "train_loss"
    )

    # ========================================================
    # ZERO-TRAINING SAFETY CHECK
    # ========================================================

    if train_loss is None:

        raise RuntimeError(
            "No train_loss was produced."
        )

    if train_loss == 0:

        raise RuntimeError(
            "Training produced "
            "train_loss = 0. "
            "This is not accepted as "
            "a valid SFT run."
        )

    # ========================================================
    # SAVE ADAPTER
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "SAVING LoRA ADAPTER"
    )

    print(
        "=" * 70
    )

    trainer.save_model(
        str(
            OUTPUT_DIR
        )
    )

    processor.save_pretrained(
        str(
            OUTPUT_DIR
        )
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_file = (
        OUTPUT_DIR
        / "training_metrics.json"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # ========================================================
    # MEMORY
    # ========================================================

    print_gpu_memory(
        "After training"
    )

    # ========================================================
    # CLEANUP
    # ========================================================

    del trainer

    del model

    del processor

    gc.collect()

    torch.cuda.empty_cache()

    print_gpu_memory(
        "After cleanup"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "SFT RUN FINISHED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "LoRA adapter:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Metrics:"
    )

    print(
        metrics_file
    )

    print()

    print(
        "Next step:"
    )

    print(
        "Evaluate the adapter on the "
        "untouched test set."
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()