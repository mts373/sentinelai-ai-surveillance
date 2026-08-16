import os
import json
import gc
import re
import random
from pathlib import Path
from collections import Counter

# Reduce CUDA allocator fragmentation.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import cv2
import numpy as np
import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# SENTINELAI
# FRESH HELD-OUT EVALUATION
#
# Compare:
#
#   1. Original LoRA
#   2. Corrective v3 LoRA
#
# IMPORTANT:
#
# Human labels were used for the 31 corrective examples.
# Therefore, every VIDEO appearing in those 31 corrections
# is completely excluded from this fresh evaluation.
#
# This prevents video-level leakage.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\SentinelAI_Qwen"
)

MODEL_NAME = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

BASELINE_ADAPTER = (
    ROOT
    / "models"
    / "sentinelai_qwen25vl_lora"
)

CORRECTIVE_ADAPTER = (
    ROOT
    / "models"
    / "sentinelai_qwen25vl_lora_corrective_v3"
)

HUMAN_LABELS_FILE = (
    ROOT
    / "dataset"
    / "test_evaluation"
    / "human_temporal_labels.json"
)

TARGETED_ERROR_FILE = (
    ROOT
    / "dataset"
    / "sft"
    / "targeted_error_sft.json"
)

OUTPUT_DIR = (
    ROOT
    / "dataset"
    / "test_evaluation"
    / "fresh_holdout"
)

HOLDOUT_FILE = (
    OUTPUT_DIR
    / "fresh_holdout_labels.json"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "fresh_holdout_comparison.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

CLASSES = [
    "Normal",
    "Fire",
    "Fight",
    "Road Accident",
]

# Maximum number of windows per class.
#
# Fire only has 8 human-labelled samples in the current
# 265-window set, so we use all available eligible Fire
# samples if fewer than 8 remain after exclusion.
#
# This gives approximately:
#
# Normal         10
# Fire            8
# Fight          10
# Road Accident  10
#
# Maximum total = 38 windows.
MAX_PER_CLASS = {
    "Normal": 10,
    "Fire": 8,
    "Fight": 10,
    "Road Accident": 10,
}

RANDOM_SEED = 42

# Same evaluation sampling used by your existing evaluator.
SAMPLE_FPS = 2.0

MAX_NEW_TOKENS = 150

# Keep evaluation memory conservative.
MIN_PIXELS = 100352
MAX_PIXELS = 401408

PROMPT = (
    "Analyze this surveillance video clip. "
    "Classify the scene as exactly one of: "
    "Normal, Fire, Fight, or Road Accident. "
    "Return ONLY a JSON object with the fields "
    "classification, evidence, and incident_summary."
)


# ============================================================
# HELPERS
# ============================================================

def normalize_video_path(path):
    """
    Normalize Windows paths so comparison is reliable.
    """

    return os.path.normcase(
        os.path.normpath(
            str(path)
        )
    )


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
# LOAD HUMAN LABELS
# ============================================================

def load_human_labels():

    print()
    print("=" * 70)
    print("LOADING HUMAN TEMPORAL LABELS")
    print("=" * 70)

    if not HUMAN_LABELS_FILE.exists():

        raise FileNotFoundError(
            f"Human labels not found:\n"
            f"{HUMAN_LABELS_FILE}"
        )

    with open(
        HUMAN_LABELS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):

        raise ValueError(
            "human_temporal_labels.json "
            "must contain a JSON list."
        )

    valid = []

    for item in data:

        label = item.get(
            "human_label"
        )

        if label not in CLASSES:
            continue

        if not all(
            key in item
            for key in [
                "video_path",
                "start",
                "end",
            ]
        ):
            continue

        video_path = Path(
            item["video_path"]
        )

        if not video_path.exists():
            print(
                "WARNING: video does not exist:"
            )
            print(
                video_path
            )
            continue

        start = float(
            item["start"]
        )

        end = float(
            item["end"]
        )

        if end <= start:
            continue

        valid.append(
            {
                "video_path": str(
                    video_path
                ),
                "source_label": item.get(
                    "source_label"
                ),
                "start": start,
                "end": end,
                "duration": item.get(
                    "duration"
                ),
                "human_label": label,
            }
        )

    print(
        f"Total annotations: {len(data)}"
    )

    print(
        f"Valid annotations: {len(valid)}"
    )

    counts = Counter(
        item["human_label"]
        for item in valid
    )

    print()
    print(
        "Human-label distribution:"
    )

    for label in CLASSES:

        print(
            f"  {label:<15}"
            f"{counts[label]}"
        )

    if not valid:

        raise RuntimeError(
            "No valid human annotations found."
        )

    return valid


# ============================================================
# LOAD TRAINING ERROR VIDEOS
# ============================================================

def load_excluded_videos():

    print()
    print("=" * 70)
    print("LOADING CORRECTIVE TRAINING VIDEOS")
    print("=" * 70)

    if not TARGETED_ERROR_FILE.exists():

        raise FileNotFoundError(
            f"Targeted error file not found:\n"
            f"{TARGETED_ERROR_FILE}"
        )

    with open(
        TARGETED_ERROR_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):

        raise ValueError(
            "targeted_error_sft.json "
            "must contain a JSON list."
        )

    excluded = set()

    for item in data:

        video = (
            item.get("video")
            or item.get("video_path")
        )

        if video:

            excluded.add(
                normalize_video_path(
                    video
                )
            )

    print(
        f"Corrective records: {len(data)}"
    )

    print(
        f"Unique videos excluded: "
        f"{len(excluded)}"
    )

    print()
    print(
        "These videos will NOT appear "
        "in the fresh holdout."
    )

    return excluded


# ============================================================
# BUILD FRESH HOLDOUT
# ============================================================

def build_fresh_holdout(
    labels,
    excluded_videos,
):

    print()
    print("=" * 70)
    print("BUILDING FRESH HELD-OUT SET")
    print("=" * 70)

    # --------------------------------------------------------
    # Remove every video used by corrective training.
    # --------------------------------------------------------

    eligible = []

    excluded_count = 0

    for item in labels:

        key = normalize_video_path(
            item["video_path"]
        )

        if key in excluded_videos:

            excluded_count += 1
            continue

        eligible.append(
            item
        )

    print(
        f"Human annotations before exclusion: "
        f"{len(labels)}"
    )

    print(
        f"Annotations removed because their "
        f"video was used for correction: "
        f"{excluded_count}"
    )

    print(
        f"Eligible fresh annotations: "
        f"{len(eligible)}"
    )

    # --------------------------------------------------------
    # Group by human ground truth.
    # --------------------------------------------------------

    grouped = {
        label: []
        for label in CLASSES
    }

    for item in eligible:

        grouped[
            item["human_label"]
        ].append(item)

    print()
    print(
        "Eligible class distribution:"
    )

    for label in CLASSES:

        print(
            f"  {label:<15}"
            f"{len(grouped[label])}"
        )

    # --------------------------------------------------------
    # Deterministic random selection.
    # --------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    selected = []

    for label in CLASSES:

        candidates = list(
            grouped[label]
        )

        rng.shuffle(
            candidates
        )

        target = min(
            MAX_PER_CLASS[label],
            len(candidates),
        )

        chosen = candidates[
            :target
        ]

        selected.extend(
            chosen
        )

    # --------------------------------------------------------
    # Sort for reproducibility.
    # --------------------------------------------------------

    selected.sort(
        key=lambda x: (
            x["human_label"],
            normalize_video_path(
                x["video_path"]
            ),
            x["start"],
        )
    )

    if not selected:

        raise RuntimeError(
            "Fresh holdout is empty."
        )

    # --------------------------------------------------------
    # Distribution.
    # --------------------------------------------------------

    counts = Counter(
        item["human_label"]
        for item in selected
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FRESH HOLDOUT DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    for label in CLASSES:

        print(
            f"  {label:<15}"
            f"{counts[label]}"
        )

    print()
    print(
        f"Fresh holdout windows: "
        f"{len(selected)}"
    )

    # --------------------------------------------------------
    # Save holdout.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        HOLDOUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            selected,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Fresh holdout saved:"
    )

    print(
        HOLDOUT_FILE
    )

    return selected


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    adapter_path
):

    if not adapter_path.exists():

        raise FileNotFoundError(
            f"LoRA adapter not found:\n"
            f"{adapter_path}"
        )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16
        ),
    )

    print()
    print(
        f"Loading adapter:"
    )

    print(
        adapter_path
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            quantization_config=bnb,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    )

    print(
        "Base model: OK"
    )

    model = (
        PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )
    )

    model.eval()

    print(
        "LoRA adapter: OK"
    )

    return model


# ============================================================
# LOAD PROCESSOR
# ============================================================

def load_processor():

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_NAME,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
    )

    if (
        processor.tokenizer.pad_token
        is None
    ):

        processor.tokenizer.pad_token = (
            processor.tokenizer.eos_token
        )

    print(
        "Processor: OK"
    )

    return processor


# ============================================================
# READ EXACT VIDEO WINDOW
# ============================================================

def read_window(
    path,
    start,
    end,
):

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video:\n"
            f"{path}"
        )

    fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 30.0
    )

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total / fps
        if total
        else 0
    )

    start = max(
        0.0,
        float(start),
    )

    end = min(
        float(end),
        duration,
    )

    if end <= start:

        cap.release()

        raise RuntimeError(
            f"Invalid window: "
            f"{start} -> {end}"
        )

    # --------------------------------------------------------
    # Sequential reading is more reliable for UCF-Crime.
    # --------------------------------------------------------

    first_frame = int(
        np.floor(
            start * fps
        )
    )

    last_frame = int(
        np.ceil(
            end * fps
        )
    )

    frames = []

    frame_index = 0

    while frame_index < last_frame:

        ok, frame = cap.read()

        if not ok:
            break

        if frame_index >= first_frame:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frames.append(
                frame
            )

        frame_index += 1

    cap.release()

    if not frames:

        raise RuntimeError(
            f"No frames extracted:\n"
            f"{path}\n"
            f"{start} -> {end}"
        )

    # --------------------------------------------------------
    # Temporal sampling.
    # --------------------------------------------------------

    target_frames = max(
        1,
        int(
            round(
                (end - start)
                * SAMPLE_FPS
            )
        ),
    )

    if len(frames) > target_frames:

        indices = np.linspace(
            0,
            len(frames) - 1,
            target_frames,
        ).astype(int)

        frames = [
            frames[i]
            for i in indices
        ]

    video = torch.from_numpy(
        np.stack(frames)
    ).permute(
        0,
        3,
        1,
        2,
    ).contiguous()

    return (
        video,
        fps,
        duration,
    )


# ============================================================
# PARSE QWEN OUTPUT
# ============================================================

def parse_prediction(
    text
):

    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip()

    # --------------------------------------------------------
    # Direct JSON.
    # --------------------------------------------------------

    try:

        data = json.loads(
            text
        )

        label = data.get(
            "classification"
        )

        if label in CLASSES:
            return label

    except Exception:
        pass

    # --------------------------------------------------------
    # JSON embedded inside text.
    # --------------------------------------------------------

    match = re.search(
        r"\{.*?\}",
        text,
        flags=re.DOTALL,
    )

    if match:

        try:

            data = json.loads(
                match.group()
            )

            label = data.get(
                "classification"
            )

            if label in CLASSES:
                return label

        except Exception:
            pass

    # --------------------------------------------------------
    # Fallback label search.
    # --------------------------------------------------------

    for label in CLASSES:

        if re.search(
            rf"\b{re.escape(label)}\b",
            text,
            flags=re.IGNORECASE,
        ):

            return label

    return None


# ============================================================
# PREDICT ONE WINDOW
# ============================================================

@torch.inference_mode()
def predict(
    model,
    processor,
    video,
):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": (
                        "temporal_window.mp4"
                    ),
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
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
        videos=[video],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: (
            value.to("cuda")
            if torch.is_tensor(value)
            else value
        )
        for key, value in inputs.items()
    }

    generated = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    prompt_length = (
        inputs["input_ids"]
        .shape[-1]
    )

    generated_text = (
        processor.batch_decode(
            generated[
                :,
                prompt_length:
            ],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    )

    prediction = parse_prediction(
        generated_text
    )

    return (
        prediction,
        generated_text,
    )


# ============================================================
# EVALUATE ONE ADAPTER
# ============================================================

def evaluate_adapter(
    adapter_name,
    adapter_path,
    holdout,
):

    print()
    print("=" * 70)
    print(
        f"EVALUATING: {adapter_name}"
    )
    print("=" * 70)

    model = load_model(
        adapter_path
    )

    processor = load_processor()

    results = []

    total = len(
        holdout
    )

    for index, item in enumerate(
        holdout,
        start=1,
    ):

        path = Path(
            item["video_path"]
        )

        start = float(
            item["start"]
        )

        end = float(
            item["end"]
        )

        truth = item[
            "human_label"
        ]

        print()
        print(
            f"[{index}/{total}] "
            f"{path.name}"
        )

        print(
            f"Window: "
            f"{start:.2f}s → "
            f"{end:.2f}s"
        )

        print(
            f"Human ground truth: "
            f"{truth}"
        )

        record = {
            "adapter": adapter_name,
            "video_path": str(path),
            "source_label": item.get(
                "source_label"
            ),
            "start": start,
            "end": end,
            "ground_truth": truth,
            "prediction": None,
            "raw_output": None,
        }

        try:

            video, fps, duration = (
                read_window(
                    path,
                    start,
                    end,
                )
            )

            print(
                f"Source FPS: "
                f"{fps:.3f}"
            )

            print(
                f"Frames sent to Qwen: "
                f"{video.shape[0]}"
            )

            prediction, raw = predict(
                model,
                processor,
                video,
            )

            record[
                "prediction"
            ] = prediction

            record[
                "raw_output"
            ] = raw

            print(
                f"Predicted: "
                f"{prediction}"
            )

            del video

            torch.cuda.empty_cache()

        except Exception as exc:

            record[
                "error"
            ] = repr(exc)

            print(
                "ERROR:"
            )

            print(
                repr(exc)
            )

            torch.cuda.empty_cache()

        results.append(
            record
        )

    # --------------------------------------------------------
    # Metrics.
    # --------------------------------------------------------

    evaluated = [
        item
        for item in results
        if item["prediction"]
        in CLASSES
    ]

    correct = sum(
        item["prediction"]
        == item["ground_truth"]
        for item in evaluated
    )

    accuracy = (
        correct / len(evaluated)
        if evaluated
        else 0.0
    )

    confusion = {
        actual: {
            predicted: 0
            for predicted in CLASSES
        }
        for actual in CLASSES
    }

    for item in evaluated:

        confusion[
            item["ground_truth"]
        ][
            item["prediction"]
        ] += 1

    per_class = {}

    for label in CLASSES:

        class_items = [
            item
            for item in evaluated
            if item["ground_truth"]
            == label
        ]

        class_correct = sum(
            item["prediction"]
            == label
            for item in class_items
        )

        class_accuracy = (
            class_correct
            / len(class_items)
            if class_items
            else 0.0
        )

        per_class[
            label
        ] = {
            "samples": len(
                class_items
            ),
            "correct": class_correct,
            "accuracy": class_accuracy,
            "accuracy_percent":
                class_accuracy * 100,
        }

    print()
    print(
        "=" * 70
    )

    print(
        f"{adapter_name} RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Evaluated: "
        f"{len(evaluated)}"
    )

    print(
        f"Correct: "
        f"{correct}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Errors/unparsed: "
        f"{len(results) - len(evaluated)}"
    )

    print()
    print(
        "Per-class:"
    )

    for label in CLASSES:

        stats = per_class[
            label
        ]

        print(
            f"  {label:<15}"
            f"{stats['samples']:>4} samples  "
            f"{stats['correct']:>4} correct  "
            f"{stats['accuracy_percent']:>6.2f}%"
        )

    # --------------------------------------------------------
    # Free GPU memory before next adapter.
    # --------------------------------------------------------

    del model
    del processor

    gc.collect()

    torch.cuda.empty_cache()

    if torch.cuda.is_available():

        torch.cuda.synchronize()

    print_gpu_memory(
        "After adapter cleanup"
    )

    return {
        "adapter_name": adapter_name,
        "adapter_path": str(
            adapter_path
        ),
        "total_holdout": len(
            holdout
        ),
        "evaluated": len(
            evaluated
        ),
        "correct": correct,
        "errors_or_unparsed":
            len(results) - len(evaluated),
        "accuracy": accuracy,
        "accuracy_percent":
            accuracy * 100,
        "confusion_matrix":
            confusion,
        "per_class":
            per_class,
        "results":
            results,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SENTINELAI - FRESH HELD-OUT "
        "ADAPTER COMPARISON"
    )
    print("=" * 70)

    print()
    print(
        "This is EVALUATION ONLY."
    )

    print(
        "NO TRAINING will be performed."
    )

    print()
    print(
        f"Baseline adapter:"
    )

    print(
        BASELINE_ADAPTER
    )

    print()
    print(
        f"Corrective v3 adapter:"
    )

    print(
        CORRECTIVE_ADAPTER
    )

    # ========================================================
    # GPU
    # ========================================================

    print()
    print("=" * 70)
    print("GPU")
    print("=" * 70)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        "CUDA: True"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "GPU memory:",
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    labels = load_human_labels()

    excluded = (
        load_excluded_videos()
    )

    holdout = build_fresh_holdout(
        labels,
        excluded,
    )

    # ========================================================
    # EVALUATE BASELINE
    # ========================================================

    baseline_results = (
        evaluate_adapter(
            "Original LoRA",
            BASELINE_ADAPTER,
            holdout,
        )
    )

    # ========================================================
    # EVALUATE CORRECTIVE V3
    # ========================================================

    corrective_results = (
        evaluate_adapter(
            "Corrective v3",
            CORRECTIVE_ADAPTER,
            holdout,
        )
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    baseline_accuracy = (
        baseline_results[
            "accuracy_percent"
        ]
    )

    corrective_accuracy = (
        corrective_results[
            "accuracy_percent"
        ]
    )

    improvement = (
        corrective_accuracy
        - baseline_accuracy
    )

    print()
    print("=" * 70)
    print(
        "FRESH HOLDOUT COMPARISON"
    )
    print("=" * 70)

    print()

    print(
        f"Fresh holdout windows: "
        f"{len(holdout)}"
    )

    print()

    print(
        f"Original LoRA:"
        f"      {baseline_accuracy:.2f}%"
    )

    print(
        f"Corrective v3:"
        f"      {corrective_accuracy:.2f}%"
    )

    print()

    print(
        f"Improvement:"
        f"          {improvement:+.2f} percentage points"
    )

    print()

    if improvement > 0:

        print(
            "RESULT: Corrective v3 "
            "performed better."
        )

    elif improvement < 0:

        print(
            "RESULT: Original LoRA "
            "performed better."
        )

    else:

        print(
            "RESULT: Both adapters "
            "performed identically."
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    comparison = {
        "experiment":
            "fresh_video_level_holdout",
        "model":
            MODEL_NAME,
        "random_seed":
            RANDOM_SEED,
        "holdout_policy": {
            "excluded_corrective_training_videos":
                len(excluded),
            "max_per_class":
                MAX_PER_CLASS,
            "total_holdout":
                len(holdout),
        },
        "holdout_file":
            str(HOLDOUT_FILE),
        "baseline":
            baseline_results,
        "corrective_v3":
            corrective_results,
        "comparison": {
            "baseline_accuracy_percent":
                baseline_accuracy,
            "corrective_accuracy_percent":
                corrective_accuracy,
            "improvement_percentage_points":
                improvement,
        },
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            comparison,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print(
        "EVALUATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Holdout:"
    )

    print(
        HOLDOUT_FILE
    )

    print()
    print(
        "Comparison:"
    )

    print(
        RESULTS_FILE
    )

    print()
    print(
        "DO NOT RETRAIN YET."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()