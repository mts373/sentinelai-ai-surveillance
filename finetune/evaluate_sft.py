import json
import gc
import re
from pathlib import Path
from collections import Counter, defaultdict

import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

from peft import PeftModel

from qwen_vl_utils import process_vision_info


# ============================================================
# SENTINELAI
# QWEN2.5-VL 7B LoRA EVALUATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\SentinelAI_Qwen"
)

MODEL_NAME = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

ADAPTER_DIR = (
    PROJECT_ROOT
    / "models"
    / "sentinelai_qwen25vl_lora"
)

TEST_MANIFEST = (
    PROJECT_ROOT
    / "dataset"
    / "clips"
    / "test.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "test_evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "test_results.json"
)


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

MIN_PIXELS = 200704

MAX_PIXELS = 602112

MAX_NEW_TOKENS = 128

# IMPORTANT:
# Start with 5.
# After the smoke test succeeds, change to None.
MAX_WINDOWS = None


PROMPT = (
    "Analyze this surveillance video clip. "
    "Classify the scene as exactly one of: "
    "Normal, Fire, Fight, or Road Accident. "
    "Return ONLY a JSON object with the fields "
    "classification, evidence, and incident_summary."
)


LABELS = [
    "Normal",
    "Fire",
    "Fight",
    "Road Accident",
]


# ============================================================
# PRINT HEADER
# ============================================================

def header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD TEST MANIFEST
# ============================================================

def load_test_manifest():

    header(
        "LOADING TEST TEMPORAL MANIFEST"
    )

    if not TEST_MANIFEST.exists():

        raise FileNotFoundError(
            f"Test manifest does not exist:\n"
            f"{TEST_MANIFEST}"
        )

    records = []

    with open(
        TEST_MANIFEST,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    print(
        f"Manifest windows: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Validate records
    # --------------------------------------------------------

    for i, record in enumerate(records):

        required = [
            "video_path",
            "label",
            "split",
            "start",
            "end",
            "duration",
        ]

        for key in required:

            if key not in record:

                raise ValueError(
                    f"Record {i} missing "
                    f"field: {key}"
                )

        video_path = Path(
            record["video_path"]
        )

        if not video_path.exists():

            raise FileNotFoundError(
                f"\nTest video does not exist:\n"
                f"{video_path}"
            )

        if record["split"] != "test":

            raise ValueError(
                f"Non-test record found:\n"
                f"{record}"
            )

        if record["label"] not in LABELS:

            raise ValueError(
                f"Invalid label:\n"
                f"{record['label']}"
            )

    print(
        "Manifest validation: OK"
    )

    return records


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    header(
        "LOADING QWEN2.5-VL + LoRA"
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        f"GPU: "
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

    if not ADAPTER_DIR.exists():

        raise FileNotFoundError(
            f"LoRA adapter does not exist:\n"
            f"{ADAPTER_DIR}"
        )

    # --------------------------------------------------------
    # Same 4-bit configuration as training.
    # --------------------------------------------------------

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

    print()
    print(
        "Loading base model..."
    )

    base_model = (
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

    print(
        "Base model: OK"
    )

    print()
    print(
        "Loading LoRA adapter..."
    )

    model = (
        PeftModel.from_pretrained(

            base_model,

            str(ADAPTER_DIR),

            is_trainable=False,

        )
    )

    model.eval()

    # --------------------------------------------------------
    # Evaluation mode.
    # --------------------------------------------------------

    model.config.use_cache = True

    print(
        "LoRA adapter: OK"
    )

    return model


# ============================================================
# LOAD PROCESSOR
# ============================================================

def load_processor():

    header(
        "LOADING PROCESSOR"
    )

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
# BUILD QWEN MESSAGE
# ============================================================

def build_messages(record):

    video_path = Path(
        record["video_path"]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Use the NORMAL Windows path.
    #
    # DO NOT convert to:
    # file:///C:/...
    #
    # Your Decord setup successfully reads
    # C:\... paths during training.
    # --------------------------------------------------------

    video_content = {

        "type": "video",

        "video": str(
            video_path
        ),

        "video_start": float(
            record["start"]
        ),

        "video_end": float(
            record["end"]
        ),

    }

    text_content = {

        "type": "text",

        "text": PROMPT,

    }

    messages = [

        {

            "role": "user",

            "content": [

                video_content,

                text_content,

            ],

        }

    ]

    return messages


# ============================================================
# NORMALIZE LABEL
# ============================================================

def normalize_label(value):

    if not isinstance(
        value,
        str,
    ):

        return None

    value = value.strip()

    # Exact match.
    if value in LABELS:

        return value

    # Case-insensitive match.
    for label in LABELS:

        if value.lower() == label.lower():

            return label

    # Remove common punctuation.
    cleaned = value.rstrip(
        ".,:;!?\"'"
    ).strip()

    for label in LABELS:

        if (
            cleaned.lower()
            == label.lower()
        ):

            return label

    return None


# ============================================================
# PARSE CLASSIFICATION
# ============================================================

def parse_classification(
    generated_text
):

    text = generated_text.strip()

    # --------------------------------------------------------
    # 1. Direct JSON.
    # --------------------------------------------------------

    try:

        obj = json.loads(
            text
        )

        if isinstance(
            obj,
            dict,
        ):

            label = normalize_label(
                obj.get(
                    "classification"
                )
            )

            if label:

                return label

    except Exception:
        pass

    # --------------------------------------------------------
    # 2. JSON embedded in extra text.
    # --------------------------------------------------------

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if match:

        try:

            obj = json.loads(
                match.group(0)
            )

            if isinstance(
                obj,
                dict,
            ):

                label = normalize_label(
                    obj.get(
                        "classification"
                    )
                )

                if label:

                    return label

        except Exception:
            pass

    # --------------------------------------------------------
    # 3. Last-resort label detection.
    # --------------------------------------------------------

    lower = text.lower()

    # Road Accident before individual words.
    search_order = [
        "Road Accident",
        "Normal",
        "Fire",
        "Fight",
    ]

    for label in search_order:

        if (
            label.lower()
            in lower
        ):

            return label

    return None


# ============================================================
# SINGLE WINDOW PREDICTION
# ============================================================

@torch.inference_mode()
def predict_window(
    model,
    processor,
    record,
):

    messages = build_messages(
        record
    )

    # --------------------------------------------------------
    # Chat template.
    # --------------------------------------------------------

    text = (
        processor.apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True,

        )
    )

    # --------------------------------------------------------
    # CRITICAL VIDEO PIPELINE
    #
    # Same working approach as training.
    #
    # NO:
    # return_video_kwargs=True
    #
    # NO:
    # video_kwargs
    # --------------------------------------------------------

    (
        image_inputs,
        video_inputs,
    ) = process_vision_info(
        messages
    )

    if video_inputs is None:

        raise RuntimeError(
            "Video decoding returned None."
        )

    # --------------------------------------------------------
    # Processor.
    # --------------------------------------------------------

    inputs = processor(

        text=[
            text
        ],

        images=image_inputs,

        videos=video_inputs,

        padding=True,

        return_tensors="pt",

    )

    # --------------------------------------------------------
    # Move tensors to GPU.
    # --------------------------------------------------------

    gpu_inputs = {}

    for key, value in inputs.items():

        if torch.is_tensor(value):

            gpu_inputs[key] = (
                value.cuda()
            )

        else:

            gpu_inputs[key] = value

    # --------------------------------------------------------
    # Generate.
    # --------------------------------------------------------

    generated_ids = model.generate(

        **gpu_inputs,

        max_new_tokens=MAX_NEW_TOKENS,

        do_sample=False,

        use_cache=True,

    )

    # --------------------------------------------------------
    # Remove prompt tokens.
    # --------------------------------------------------------

    generated_ids_trimmed = []

    for input_ids, output_ids in zip(

        gpu_inputs["input_ids"],

        generated_ids,

    ):

        generated_ids_trimmed.append(

            output_ids[
                input_ids.shape[-1]:
            ]

        )

    # --------------------------------------------------------
    # Decode.
    # --------------------------------------------------------

    generated_text = (
        processor.batch_decode(

            generated_ids_trimmed,

            skip_special_tokens=True,

            clean_up_tokenization_spaces=False,

        )[0]
    )

    predicted_label = (
        parse_classification(
            generated_text
        )
    )

    return (
        predicted_label,
        generated_text,
    )


# ============================================================
# WINDOW METRICS
# ============================================================

def calculate_window_metrics(
    results
):

    valid = [

        r for r in results

        if r["prediction"]
        in LABELS

    ]

    if not valid:

        return {
            "accuracy": None,
            "evaluated": 0,
            "unparsed": len(results),
        }

    correct = sum(

        r["ground_truth"]
        == r["prediction"]

        for r in valid

    )

    accuracy = (
        correct
        / len(valid)
    )

    return {

        "accuracy": accuracy,

        "correct": correct,

        "evaluated": len(valid),

        "unparsed":
            len(results)
            - len(valid),

    }


# ============================================================
# VIDEO-LEVEL MAJORITY VOTE
# ============================================================

def aggregate_videos(
    results
):

    grouped = defaultdict(list)

    for result in results:

        grouped[
            result["video_path"]
        ].append(result)

    video_results = []

    for video_path, windows in (
        grouped.items()
    ):

        ground_truth = (
            windows[0]["ground_truth"]
        )

        valid_predictions = [

            w["prediction"]

            for w in windows

            if w["prediction"]
            in LABELS

        ]

        counts = Counter(
            valid_predictions
        )

        prediction = None

        if counts:

            max_count = max(
                counts.values()
            )

            winners = [

                label

                for label, count
                in counts.items()

                if count == max_count

            ]

            # Deterministic tie-break:
            # first chronological prediction.
            for window in windows:

                if (
                    window["prediction"]
                    in winners
                ):

                    prediction = (
                        window["prediction"]
                    )

                    break

        video_results.append({

            "video_path":
                video_path,

            "ground_truth":
                ground_truth,

            "prediction":
                prediction,

            "correct":
                prediction == ground_truth,

            "windows":
                len(windows),

            "valid_windows":
                len(valid_predictions),

            "window_predictions":
                [
                    w["prediction"]
                    for w in windows
                ],

        })

    return video_results


# ============================================================
# VIDEO METRICS
# ============================================================

def calculate_video_metrics(
    video_results
):

    valid = [

        r for r in video_results

        if r["prediction"]
        in LABELS

    ]

    if not valid:

        return {
            "accuracy": None,
            "evaluated": 0,
            "unparsed": len(video_results),
        }

    correct = sum(

        r["correct"]

        for r in valid

    )

    return {

        "accuracy":
            correct / len(valid),

        "correct":
            correct,

        "evaluated":
            len(valid),

        "unparsed":
            len(video_results)
            - len(valid),

    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def confusion_matrix(
    results
):

    matrix = {

        actual: {

            predicted: 0

            for predicted in LABELS

        }

        for actual in LABELS

    }

    for result in results:

        actual = (
            result["ground_truth"]
        )

        predicted = (
            result["prediction"]
        )

        if (
            actual in LABELS
            and predicted in LABELS
        ):

            matrix[
                actual
            ][
                predicted
            ] += 1

    return matrix


def print_confusion_matrix(
    matrix
):

    print()

    print(
        "=" * 70
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        "=" * 70
    )

    print(
        f"{'Actual':<18}"
        f"{'Normal':>12}"
        f"{'Fire':>12}"
        f"{'Fight':>12}"
        f"{'Road Accident':>16}"
    )

    for actual in LABELS:

        print(

            f"{actual:<18}"

            f"{matrix[actual]['Normal']:>12}"

            f"{matrix[actual]['Fire']:>12}"

            f"{matrix[actual]['Fight']:>12}"

            f"{matrix[actual]['Road Accident']:>16}"

        )


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "SENTINELAI - QWEN2.5-VL LoRA EVALUATION"
    )

    print(
        f"Base model:\n"
        f"{MODEL_NAME}"
    )

    print()

    print(
        f"LoRA adapter:\n"
        f"{ADAPTER_DIR}"
    )

    print()

    print(
        f"Test manifest:\n"
        f"{TEST_MANIFEST}"
    )

    print()

    print(
        "Evaluation unit:"
    )

    print(
        "10-second temporal test windows"
    )

    print()

    print(
        "Video aggregation:"
    )

    print(
        "Majority vote across windows"
    )

    # --------------------------------------------------------
    # Load manifest.
    # --------------------------------------------------------

    records = (
        load_test_manifest()
    )

    # --------------------------------------------------------
    # Smoke-test limit.
    # --------------------------------------------------------

    if MAX_WINDOWS is not None:

        records = records[
            :MAX_WINDOWS
        ]

        print()

        print(
            f"SMOKE TEST:"
            f" evaluating only "
            f"{len(records)} windows"
        )

    # --------------------------------------------------------
    # Load model.
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load processor.
    # --------------------------------------------------------

    processor = (
        load_processor()
    )

    # --------------------------------------------------------
    # Evaluation.
    # --------------------------------------------------------

    header(
        "STARTING EVALUATION"
    )

    results = []

    for index, record in enumerate(

        records,

        start=1,

    ):

        video_name = (
            Path(
                record["video_path"]
            ).name
        )

        print()

        print(
            f"[{index}/{len(records)}] "
            f"{video_name}"
        )

        print(
            f"Window: "
            f"{float(record['start']):.2f}s"
            f" → "
            f"{float(record['end']):.2f}s"
        )

        print(
            f"Expected: "
            f"{record['label']}"
        )

        try:

            (
                prediction,
                raw_output,
            ) = predict_window(

                model,

                processor,

                record,

            )

            print(
                f"Predicted: "
                f"{prediction}"
            )

            print(
                "Raw output:"
            )

            print(
                raw_output[:500]
            )

        except Exception as exc:

            print()

            print(
                "EVALUATION ERROR:"
            )

            print(
                repr(exc)
            )

            prediction = None

            raw_output = ""

        result = {

            "video_path":
                record["video_path"],

            "clip_id":
                record.get(
                    "clip_id"
                ),

            "ground_truth":
                record["label"],

            "prediction":
                prediction,

            "raw_output":
                raw_output,

            "start":
                record["start"],

            "end":
                record["end"],

            "duration":
                record["duration"],

        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Save progress.
        # ----------------------------------------------------

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(

                {

                    "status":
                        "in_progress",

                    "results":
                        results,

                },

                f,

                indent=2,

            )

    # ========================================================
    # WINDOW RESULTS
    # ========================================================

    window_metrics = (
        calculate_window_metrics(
            results
        )
    )

    matrix = (
        confusion_matrix(
            results
        )
    )

    print_confusion_matrix(
        matrix
    )

    print()

    print(
        "=" * 70
    )

    print(
        "WINDOW RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Accuracy: "
        f"{window_metrics['accuracy']}"
    )

    print(
        f"Evaluated: "
        f"{window_metrics['evaluated']}"
    )

    print(
        f"Unparsed: "
        f"{window_metrics['unparsed']}"
    )

    # ========================================================
    # VIDEO AGGREGATION
    # ========================================================

    video_results = (
        aggregate_videos(
            results
        )
    )

    video_metrics = (
        calculate_video_metrics(
            video_results
        )
    )

    print()

    print(
        "=" * 70
    )

    print(
        "VIDEO-LEVEL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Videos represented: "
        f"{len(video_results)}"
    )

    print(
        f"Video accuracy: "
        f"{video_metrics['accuracy']}"
    )

    print(
        f"Evaluated videos: "
        f"{video_metrics['evaluated']}"
    )

    print(
        f"Unparsed videos: "
        f"{video_metrics['unparsed']}"
    )

    # ========================================================
    # PER-VIDEO RESULTS
    # ========================================================

    print()

    print(
        "VIDEO PREDICTIONS"
    )

    print(
        "-" * 70
    )

    for result in video_results:

        status = (
            "CORRECT"
            if result["correct"]
            else "WRONG"
        )

        print(

            f"{status:<8}"

            f"{Path(result['video_path']).name:<40}"

            f"Expected="
            f"{result['ground_truth']:<15}"

            f"Predicted="
            f"{result['prediction']}"

        )

    # ========================================================
    # SAVE FINAL RESULTS
    # ========================================================

    final_results = {

        "status":
            "complete",

        "base_model":
            MODEL_NAME,

        "adapter":
            str(ADAPTER_DIR),

        "test_manifest":
            str(TEST_MANIFEST),

        "max_windows":
            MAX_WINDOWS,

        "windows_evaluated":
            len(results),

        "videos_evaluated":
            len(video_results),

        "window_metrics":
            window_metrics,

        "window_confusion_matrix":
            matrix,

        "video_metrics":
            video_metrics,

        "window_results":
            results,

        "video_results":
            video_results,

    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(

            final_results,

            f,

            indent=2,

        )

    # ========================================================
    # CLEANUP
    # ========================================================

    del model

    del processor

    gc.collect()

    torch.cuda.empty_cache()

    # ========================================================
    # COMPLETE
    # ========================================================

    header(
        "EVALUATION COMPLETE"
    )

    print(
        "Results:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()