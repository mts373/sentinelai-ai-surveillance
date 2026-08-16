# ============================================================
# SENTINELAI - VIDEO INFERENCE ENGINE
# ============================================================
#
# File:
# C:\SentinelAI_Qwen\src\inference_engine.py
#
# Purpose:
#
#   Preprocessed video windows
#              ↓
#       Existing qwen_engine.py
#              ↓
#       Qwen2.5-VL + LoRA
#              ↓
#       Classification
#              ↓
#       JSON results
#
# IMPORTANT:
#
# load_model() returns:
#
#     model, processor
#
# qwen_generate_video() expects:
#
#     model
#     processor
#     video_path
#
# NO TRAINING IS PERFORMED.
# ============================================================

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    r"C:\SentinelAI_Qwen"
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR)
    )


# ============================================================
# EXISTING QWEN ENGINE
# ============================================================

from qwen_engine import (
    load_model,
    qwen_generate_video,
    parse_json_output,
)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "Normal",
    "Fire",
    "Fight",
    "Road Accident",
]


# ============================================================
# MANIFEST LOADING
# ============================================================

def load_manifest(
    manifest_path: Path
):

    manifest_path = Path(
        manifest_path
    )

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"\nManifest not found:\n"
            f"{manifest_path}"
        )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            "manifest.json must contain a JSON object."
        )

    windows = data.get(
        "windows"
    )

    if not isinstance(
        windows,
        list,
    ):

        raise ValueError(
            "manifest.json does not contain "
            "a valid 'windows' list."
        )

    if len(windows) == 0:

        raise ValueError(
            "manifest.json contains zero windows."
        )

    return data


# ============================================================
# VALIDATE WINDOW
# ============================================================

def validate_window(
    window
):

    if "clip_path" not in window:

        raise ValueError(
            "Manifest window has no clip_path."
        )

    clip_path = Path(
        window["clip_path"]
    )

    if not clip_path.exists():

        raise FileNotFoundError(
            f"\nPreprocessed clip not found:\n"
            f"{clip_path}"
        )

    start = float(
        window.get(
            "start",
            0.0,
        )
    )

    end = float(
        window.get(
            "end",
            0.0,
        )
    )

    if end <= start:

        raise ValueError(
            f"Invalid window:\n"
            f"{start} -> {end}"
        )

    return (
        clip_path,
        start,
        end,
    )


# ============================================================
# NORMALIZE CLASSIFICATION
# ============================================================

def normalize_classification(
    value
):

    if value is None:

        return None

    text = str(
        value
    ).strip()

    for label in CLASSES:

        if (
            text.lower()
            == label.lower()
        ):

            return label

    normalized = (
        text.lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    for label in CLASSES:

        label_normalized = (
            label.lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        if (
            normalized
            == label_normalized
        ):

            return label

    return None


# ============================================================
# PARSE MODEL OUTPUT
# ============================================================

def parse_prediction(
    raw_output
):

    if raw_output is None:

        return {
            "classification": None,
            "evidence": "",
            "incident_summary": "",
        }

    if not isinstance(
        raw_output,
        str,
    ):

        raw_output = str(
            raw_output
        )

    # --------------------------------------------------------
    # FIRST: USE EXISTING PROJECT PARSER
    # --------------------------------------------------------

    try:

        parsed = (
            parse_json_output(
                raw_output
            )
        )

        if isinstance(
            parsed,
            dict,
        ):

            classification = (
                normalize_classification(
                    parsed.get(
                        "classification"
                    )
                )
            )

            if classification is not None:

                return {
                    "classification":
                        classification,

                    "evidence":
                        str(
                            parsed.get(
                                "evidence",
                                "",
                            )
                        ),

                    "incident_summary":
                        str(
                            parsed.get(
                                "incident_summary",
                                "",
                            )
                        ),
                }

    except Exception:

        pass

    # --------------------------------------------------------
    # SECOND: DIRECT JSON
    # --------------------------------------------------------

    try:

        parsed = json.loads(
            raw_output
        )

        if isinstance(
            parsed,
            dict,
        ):

            classification = (
                normalize_classification(
                    parsed.get(
                        "classification"
                    )
                )
            )

            if classification is not None:

                return {
                    "classification":
                        classification,

                    "evidence":
                        str(
                            parsed.get(
                                "evidence",
                                "",
                            )
                        ),

                    "incident_summary":
                        str(
                            parsed.get(
                                "incident_summary",
                                "",
                            )
                        ),
                }

    except Exception:

        pass

    # --------------------------------------------------------
    # THIRD: FIND JSON OBJECT INSIDE OUTPUT
    # --------------------------------------------------------

    start_index = (
        raw_output.find("{")
    )

    end_index = (
        raw_output.rfind("}")
    )

    if (
        start_index >= 0
        and end_index > start_index
    ):

        candidate = raw_output[
            start_index:
            end_index + 1
        ]

        try:

            parsed = json.loads(
                candidate
            )

            if isinstance(
                parsed,
                dict,
            ):

                classification = (
                    normalize_classification(
                        parsed.get(
                            "classification"
                        )
                    )
                )

                if classification is not None:

                    return {
                        "classification":
                            classification,

                        "evidence":
                            str(
                                parsed.get(
                                    "evidence",
                                    "",
                                )
                            ),

                        "incident_summary":
                            str(
                                parsed.get(
                                    "incident_summary",
                                    "",
                                )
                            ),
                    }

        except Exception:

            pass

    # --------------------------------------------------------
    # FAILED PARSING
    # --------------------------------------------------------

    return {
        "classification": None,
        "evidence": "",
        "incident_summary": "",
    }


# ============================================================
# INFERENCE ONE VIDEO
# ============================================================

def infer_clip(
    model,
    processor,
    clip_path,
    start,
    end,
):

    clip_path = Path(
        clip_path
    )

    print()
    print(
        f"Video: {clip_path.name}"
    )

    print(
        f"Window: "
        f"{start:.2f}s -> "
        f"{end:.2f}s"
    )

    inference_start = (
        time.time()
    )

    try:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # This is the exact interface used by your existing
        # qwen_engine.py.
        # ----------------------------------------------------

        raw_output = (
            qwen_generate_video(
                model=model,
                processor=processor,
                video_path=clip_path,
            )
        )

        parsed = (
            parse_prediction(
                raw_output
            )
        )

        processing_time = (
            time.time()
            - inference_start
        )

        classification = (
            parsed[
                "classification"
            ]
        )

        print()
        print(
            f"Prediction: "
            f"{classification}"
        )

        print(
            f"Processing time: "
            f"{processing_time:.2f}s"
        )

        return {
            "clip_id":
                clip_path.stem,

            "clip_path":
                str(
                    clip_path
                ),

            "start":
                start,

            "end":
                end,

            "duration":
                end - start,

            "classification":
                classification,

            "evidence":
                parsed[
                    "evidence"
                ],

            "incident_summary":
                parsed[
                    "incident_summary"
                ],

            "raw_output":
                raw_output,

            "processing_time":
                round(
                    processing_time,
                    2,
                ),

            "error":
                None,
        }

    except Exception as error:

        processing_time = (
            time.time()
            - inference_start
        )

        print()
        print(
            "ERROR:"
        )

        print(
            repr(error)
        )

        return {
            "clip_id":
                clip_path.stem,

            "clip_path":
                str(
                    clip_path
                ),

            "start":
                start,

            "end":
                end,

            "duration":
                end - start,

            "classification":
                None,

            "evidence":
                "",

            "incident_summary":
                "",

            "raw_output":
                None,

            "processing_time":
                round(
                    processing_time,
                    2,
                ),

            "error":
                repr(error),
        }


# ============================================================
# PROCESS MANIFEST
# ============================================================

def process_manifest(
    manifest_path
):

    manifest_path = Path(
        manifest_path
    )

    manifest = load_manifest(
        manifest_path
    )

    windows = (
        manifest[
            "windows"
        ]
    )

    print()
    print("=" * 70)
    print(
        "SENTINELAI - LOADING QWEN"
    )
    print("=" * 70)

    print()
    print(
        "Loading Qwen2.5-VL + LoRA..."
    )

    # ========================================================
    # CRITICAL FIX
    #
    # load_model() RETURNS TWO OBJECTS:
    #
    #     model, processor
    #
    # ========================================================

    model, processor = (
        load_model()
    )

    print()
    print(
        "Model: OK"
    )

    print(
        "Processor: OK"
    )

    # ========================================================
    # INFERENCE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "STARTING VIDEO INFERENCE"
    )
    print("=" * 70)

    print()
    print(
        f"Windows: "
        f"{len(windows)}"
    )

    results = []

    try:

        for index, window in enumerate(
            windows,
            start=1,
        ):

            print()
            print(
                "-" * 70
            )

            print(
                f"[{index}/{len(windows)}]"
            )

            try:

                (
                    clip_path,
                    start,
                    end,
                ) = validate_window(
                    window
                )

                result = infer_clip(
                    model=model,
                    processor=processor,
                    clip_path=clip_path,
                    start=start,
                    end=end,
                )

                results.append(
                    result
                )

            except Exception as error:

                print()
                print(
                    "WINDOW ERROR:"
                )

                print(
                    repr(error)
                )

                results.append(
                    {
                        "clip_id":
                            window.get(
                                "clip_id"
                            ),

                        "clip_path":
                            window.get(
                                "clip_path"
                            ),

                        "start":
                            window.get(
                                "start"
                            ),

                        "end":
                            window.get(
                                "end"
                            ),

                        "duration":
                            window.get(
                                "duration"
                            ),

                        "classification":
                            None,

                        "evidence":
                            "",

                        "incident_summary":
                            "",

                        "raw_output":
                            None,

                        "processing_time":
                            0,

                        "error":
                            repr(error),
                    }
                )

            # ------------------------------------------------
            # Cleanup temporary Python references.
            #
            # qwen_engine.py already handles its own GPU
            # cleanup before/after inference.
            # ------------------------------------------------

            gc.collect()

    finally:

        # ====================================================
        # RELEASE MODEL
        # ====================================================

        try:

            del model

        except Exception:

            pass

        try:

            del processor

        except Exception:

            pass

        gc.collect()

        try:

            import torch

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

                try:

                    torch.cuda.ipc_collect()

                except Exception:

                    pass

        except Exception:

            pass

    return results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    manifest_path,
):

    manifest_path = Path(
        manifest_path
    )

    output_path = (
        manifest_path.parent
        / "inference_results.json"
    )

    successful = 0
    failed = 0

    for result in results:

        if (
            result.get(
                "classification"
            )
            in CLASSES
        ):

            successful += 1

        else:

            failed += 1

    output = {

        "manifest":
            str(
                manifest_path
            ),

        "total_windows":
            len(results),

        "successful_predictions":
            successful,

        "errors_or_unparsed":
            failed,

        "results":
            results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    results
):

    counts = {
        "Normal": 0,
        "Fire": 0,
        "Fight": 0,
        "Road Accident": 0,
    }

    errors = 0

    for result in results:

        prediction = (
            result.get(
                "classification"
            )
        )

        if prediction in counts:

            counts[
                prediction
            ] += 1

        else:

            errors += 1

    print()
    print("=" * 70)
    print(
        "INFERENCE SUMMARY"
    )
    print("=" * 70)

    print()

    print(
        f"Normal            "
        f"{counts['Normal']}"
    )

    print(
        f"Fire              "
        f"{counts['Fire']}"
    )

    print(
        f"Fight             "
        f"{counts['Fight']}"
    )

    print(
        f"Road Accident     "
        f"{counts['Road Accident']}"
    )

    print(
        f"ERROR / UNPARSED  "
        f"{errors}"
    )

    print()

    print(
        f"Total windows: "
        f"{len(results)}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "SentinelAI video inference "
            "using the existing Qwen engine."
        )
    )

    parser.add_argument(
        "manifest",
        type=str,
        help=(
            "Path to preprocessing manifest.json"
        ),
    )

    args = parser.parse_args()

    manifest_path = Path(
        args.manifest
    ).resolve()

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"\nManifest not found:\n"
            f"{manifest_path}"
        )

    print()
    print("=" * 70)
    print(
        "SENTINELAI - VIDEO INFERENCE"
    )
    print("=" * 70)

    print()
    print(
        "Manifest:"
    )

    print(
        manifest_path
    )

    results = process_manifest(
        manifest_path
    )

    output_path = save_results(
        results,
        manifest_path,
    )

    print_summary(
        results
    )

    print()
    print(
        "Results saved:"
    )

    print(
        output_path
    )

    print()
    print("=" * 70)
    print(
        "INFERENCE COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()