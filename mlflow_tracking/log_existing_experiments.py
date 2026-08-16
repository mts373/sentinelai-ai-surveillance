from pathlib import Path
import mlflow


# ============================================================
# SENTINELAI
# MLflow - Historical Experiment Tracking
# ============================================================
#
# This script records experiments that were completed BEFORE
# MLflow was integrated into the project.
#
# IMPORTANT:
# These runs are explicitly marked as "historical".
# We do NOT claim MLflow was running during those experiments.
#
# Future training/evaluation runs should be logged directly
# from the training/evaluation pipeline.
# ============================================================


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(r"C:\SentinelAI_Qwen")

MLFLOW_DIR = PROJECT_ROOT / "mlflow_tracking"
MLFLOW_DB = MLFLOW_DIR / "mlflow.db"

MLFLOW_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# MLflow DATABASE BACKEND
# ------------------------------------------------------------
#
# MLflow 3.x filesystem tracking is deprecated.
# Use SQLite locally.
# ------------------------------------------------------------

TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

mlflow.set_tracking_uri(TRACKING_URI)


# ------------------------------------------------------------
# EXPERIMENT
# ------------------------------------------------------------

EXPERIMENT_NAME = "SentinelAI - Qwen2.5-VL Experiments"

mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# HELPER
# ============================================================

def log_text_file(text: str, filename: str):
    """
    Log a text artifact without creating unnecessary files
    in the project directory.
    """
    mlflow.log_text(text.strip() + "\n", filename)


# ============================================================
# EXPERIMENT 1
# BASELINE QLORA / SFT
# ============================================================

def log_baseline_sft():
    print("\n[1/4] Logging baseline SFT experiment...")

    with mlflow.start_run(
        run_name="baseline-qlora-sft-historical"
    ):

        mlflow.set_tags({
            "record_type": "historical",
            "stage": "baseline",
            "model_family": "Qwen2.5-VL",
            "training_method": "QLoRA",
            "project": "SentinelAI",
        })

        mlflow.log_params({
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "method": "QLoRA",
            "dataset": "sft_train.json",
            "epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-4,
            "warmup_steps": 2,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "min_pixels": 200704,
            "max_pixels": 602112,
        })

        log_text_file(
            """
Baseline SentinelAI QLoRA/SFT experiment.

Model:
Qwen/Qwen2.5-VL-7B-Instruct

Training method:
QLoRA

The baseline fine-tuning pipeline was developed before
MLflow was introduced into the project.

This run is therefore recorded as a historical experiment.

The baseline model was subsequently evaluated using the
human-ground-truth temporal evaluation pipeline.
""",
            "baseline_experiment_notes.txt",
        )

    print("    Baseline SFT logged.")


# ============================================================
# EXPERIMENT 2
# HUMAN GROUND-TRUTH TEMPORAL EVALUATION
# ============================================================

def log_human_temporal_evaluation():
    print("\n[2/4] Logging human-ground-truth evaluation...")

    with mlflow.start_run(
        run_name="human-ground-truth-temporal-evaluation"
    ):

        mlflow.set_tags({
            "record_type": "historical",
            "stage": "evaluation",
            "evaluation_type": "human-ground-truth",
            "project": "SentinelAI",
        })

        mlflow.log_params({
            "evaluation_level": "temporal-window",
            "evaluation_samples": 265,
            "classes": (
                "Normal, Fire, Fight, Road Accident"
            ),
        })

        mlflow.log_metric(
            "accuracy_percent",
            88.30,
        )

        log_text_file(
            """
Human-ground-truth temporal evaluation.

Evaluation size:
265 temporal windows

Accuracy:
88.30%

This evaluation was used to identify model errors before
building the targeted corrective dataset.

The error analysis identified:
31 incorrect windows.

Those errors became the basis for the corrective-data
workflow.
""",
            "human_temporal_evaluation_notes.txt",
        )

    print("    Human-ground-truth evaluation logged.")


# ============================================================
# EXPERIMENT 3
# CORRECTIVE QLORA V3
# ============================================================

def log_corrective_sft():
    print("\n[3/4] Logging corrective QLoRA v3 experiment...")

    with mlflow.start_run(
        run_name="corrective-qlora-v3-historical"
    ):

        mlflow.set_tags({
            "record_type": "historical",
            "stage": "corrective-training",
            "version": "v3",
            "project": "SentinelAI",
        })

        mlflow.log_params({
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "method": "QLoRA",
            "dataset": "corrective_sft_train.json",

            "corrective_records": 139,
            "epochs": 1,

            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,

            "effective_batch_size": 8,

            "learning_rate": 5e-5,
            "warmup_steps": 2,

            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,

            "video_sampling_fps": 0.5,
            "max_video_frames": 8,

            "min_pixels": 100352,
            "max_pixels_per_frame": 401408,

            "memory_optimization": (
                "4-bit QLoRA + gradient checkpointing "
                "+ controlled video sampling"
            ),
        })

        # Exact value observed in the completed run.
        mlflow.log_metric(
            "train_loss",
            1.9483672976493835,
        )

        log_text_file(
            """
Corrective QLoRA v3 experiment.

Purpose:
Correct model errors identified through human-ground-truth
temporal evaluation.

Corrective dataset:
139 records.

Training:
1 epoch
per-device batch size = 1
gradient accumulation = 8
effective batch size = 8
learning rate = 5e-5

Video memory controls:
sampling FPS = 0.5
maximum frames = 8
maximum pixels/frame = 401408

Training loss:
1.9483672976493835

The corrective model was subsequently evaluated on a fresh
held-out video set.
""",
            "corrective_training_notes.txt",
        )

    print("    Corrective QLoRA v3 logged.")


# ============================================================
# EXPERIMENT 4
# FRESH HELD-OUT VIDEO-LEVEL EVALUATION
# ============================================================

def log_fresh_holdout():
    print("\n[4/4] Logging fresh held-out evaluation...")

    with mlflow.start_run(
        run_name="fresh-holdout-video-level-comparison"
    ):

        mlflow.set_tags({
            "record_type": "historical",
            "stage": "fresh-holdout-evaluation",
            "evaluation_type": "video-level",
            "project": "SentinelAI",
        })

        mlflow.log_params({
            "videos_evaluated": 13,
            "evaluation_level": "video",
            "models_compared": 2,
            "classes": (
                "Normal, Fire, Fight, Road Accident"
            ),
        })

        # Actual result from the completed evaluation.
        mlflow.log_metric(
            "original_lora_accuracy_percent",
            92.31,
        )

        mlflow.log_metric(
            "corrective_v3_accuracy_percent",
            92.31,
        )

        mlflow.log_metric(
            "improvement_percentage_points",
            0.00,
        )

        mlflow.log_metric(
            "original_correct_videos",
            12,
        )

        mlflow.log_metric(
            "corrective_v3_correct_videos",
            12,
        )

        # Per-class video results from the recorded evaluation.
        mlflow.log_metrics({
            "original_normal_accuracy_percent": 100.0,
            "corrective_normal_accuracy_percent": 100.0,

            "original_fire_accuracy_percent": 100.0,
            "corrective_fire_accuracy_percent": 100.0,

            "original_fight_accuracy_percent": 0.0,
            "corrective_fight_accuracy_percent": 0.0,

            "original_road_accident_accuracy_percent": 100.0,
            "corrective_road_accident_accuracy_percent": 100.0,
        })

        log_text_file(
            """
Fresh held-out video-level evaluation.

Number of videos:
13

Original LoRA:
12 correct
92.31%

Corrective v3:
12 correct
92.31%

Improvement:
0.00 percentage points

Per-class video performance:

Normal:
Original = 100.00%
Corrective = 100.00%

Fire:
Original = 100.00%
Corrective = 100.00%

Fight:
Original = 0.00%
Corrective = 0.00%

Road Accident:
Original = 100.00%
Corrective = 100.00%

No video-level disagreements were observed.

Decision:
Do not blindly continue corrective training.
The corrective experiment did not improve the fresh
video-level result, so further training should be based
on additional error analysis rather than simply increasing
the epoch count.
""",
            "fresh_holdout_notes.txt",
        )

    print("    Fresh holdout evaluation logged.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SENTINELAI - MLflow HISTORICAL EXPERIMENT TRACKING")
    print("=" * 70)

    print()
    print(f"MLflow version: {mlflow.__version__}")
    print(f"Tracking URI:   {TRACKING_URI}")
    print(f"Database:       {MLFLOW_DB}")

    log_baseline_sft()
    log_human_temporal_evaluation()
    log_corrective_sft()
    log_fresh_holdout()

    print()
    print("=" * 70)
    print("MLFLOW TRACKING COMPLETE")
    print("=" * 70)

    print()
    print("Experiment:")
    print(f"  {EXPERIMENT_NAME}")

    print()
    print("Runs created:")
    print("  1. baseline-qlora-sft-historical")
    print("  2. human-ground-truth-temporal-evaluation")
    print("  3. corrective-qlora-v3-historical")
    print("  4. fresh-holdout-video-level-comparison")

    print()
    print("Start MLflow UI with:")
    print(
        '  mlflow ui --backend-store-uri '
        f'"{TRACKING_URI}"'
    )

    print()
    print("Open:")
    print("  http://127.0.0.1:5000")


if __name__ == "__main__":
    main()