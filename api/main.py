import json
import os
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# SENTINELAI - FASTAPI BACKEND
# ============================================================
#
# Lovable Frontend
#       ↓
# POST /api/analyze-video
#       ↓
# video_preprocessor.py
#       ↓
# manifest.json
#       ↓
# inference_engine.py
#       ↓
# inference_results.json
#       ↓
# FastAPI
#       ↓
# Lovable
#
# IMPORTANT:
# This API does NOT duplicate Qwen inference.
# It launches the already-tested SentinelAI pipeline.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\SentinelAI_Qwen")

PYTHON_EXE = (
    PROJECT_ROOT
    / ".venv"
    / "Scripts"
    / "python.exe"
)

PREPROCESSOR = (
    PROJECT_ROOT
    / "src"
    / "video_preprocessor.py"
)

INFERENCE_ENGINE = (
    PROJECT_ROOT
    / "src"
    / "inference_engine.py"
)

UPLOAD_ROOT = (
    PROJECT_ROOT
    / "api"
    / "uploads"
)

JOB_ROOT = (
    PROJECT_ROOT
    / "api"
    / "jobs"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_UPLOAD_BYTES = 500 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}

CLASSES = (
    "Normal",
    "Fire",
    "Fight",
    "Road Accident",
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

UPLOAD_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

JOB_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SentinelAI API",
    version="1.0.0",
    description=(
        "FastAPI backend for SentinelAI "
        "Qwen2.5-VL surveillance analysis."
    ),
)


# ============================================================
# CORS
# ============================================================
#
# Required because Lovable frontend runs on another origin.
#
# For local MVP:
# allow all origins.
#
# Later, restrict this to your deployed frontend domain.
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RUNTIME STATE
# ============================================================

JOBS: dict[str, dict[str, Any]] = {}

INCIDENTS: dict[str, dict[str, Any]] = {}

JOBS_LOCK = threading.Lock()

# CRITICAL:
# Never allow two Qwen inference processes to run
# simultaneously on the same GPU.
GPU_LOCK = threading.Lock()


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def set_job(
    job_id: str,
    **updates: Any,
) -> None:

    with JOBS_LOCK:

        if job_id in JOBS:
            JOBS[job_id].update(
                updates
            )


def load_json(
    path: Path,
) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# GPU STATUS
# ============================================================

def gpu_status() -> dict[str, Any]:

    if not torch.cuda.is_available():

        return {
            "available": False,
            "name": None,
            "utilization_percent": None,
            "memory_used_gb": None,
            "memory_total_gb": None,
        }

    properties = (
        torch.cuda.get_device_properties(0)
    )

    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "utilization_percent": None,
        "memory_used_gb": round(
            torch.cuda.memory_allocated(0)
            / (1024 ** 3),
            2,
        ),
        "memory_total_gb": round(
            properties.total_memory
            / (1024 ** 3),
            2,
        ),
    }


# ============================================================
# FIND MANIFEST
# ============================================================

def find_manifest(
    preprocessed_dir: Path,
) -> Path:

    manifest = (
        preprocessed_dir
        / "manifest.json"
    )

    if manifest.exists():
        return manifest

    candidates = sorted(
        preprocessed_dir.glob(
            "*.json"
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        "manifest.json was not found after "
        "video preprocessing.\n\n"
        f"Directory:\n{preprocessed_dir}"
    )


# ============================================================
# FIND INFERENCE RESULTS
# ============================================================

def find_inference_results(
    preprocessed_dir: Path,
) -> Path:

    preferred = (
        preprocessed_dir
        / "inference_results.json"
    )

    if preferred.exists():
        return preferred

    candidates = sorted(
        preprocessed_dir.glob(
            "*inference*.json"
        )
    )

    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        "inference_results.json was not found "
        "after Qwen inference.\n\n"
        f"Directory:\n{preprocessed_dir}"
    )


# ============================================================
# RUN CHILD PROCESS
# ============================================================

def run_process(
    command: list[str],
    job_id: str,
    stage: str,
) -> None:

    set_job(
        job_id,
        stage=stage,
        message=(
            f"Running {stage}..."
        ),
    )

    creationflags = 0

    if os.name == "nt":

        creationflags = (
            subprocess.CREATE_NO_WINDOW
        )

    # --------------------------------------------------------
    # WINDOWS UTF-8 FIX
    # --------------------------------------------------------
    #
    # Your previous FastAPI test failed inside:
    #
    # qwen_engine.py
    #
    # with:
    #
    # UnicodeEncodeError: 'charmap' codec can't encode...
    #
    # Force the child Python process to use UTF-8.
    # --------------------------------------------------------

    env = os.environ.copy()

    env[
        "PYTHONUTF8"
    ] = "1"

    env[
        "PYTHONIOENCODING"
    ] = "utf-8"

    env[
        "PYTHONLEGACYWINDOWSSTDIO"
    ] = "0"

    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),

        stdout=subprocess.PIPE,

        stderr=subprocess.STDOUT,

        text=True,

        encoding="utf-8",

        errors="replace",

        bufsize=1,

        creationflags=creationflags,

        env=env,
    )

    output_lines: list[str] = []

    if process.stdout is None:

        raise RuntimeError(
            "Unable to capture child process output."
        )

    for line in process.stdout:

        line = line.rstrip()

        if not line:
            continue

        output_lines.append(
            line
        )

        # Keep latest backend log available
        # through GET /api/analyze-video/{id}

        set_job(
            job_id,
            last_log=line,
        )

    return_code = (
        process.wait()
    )

    if return_code != 0:

        tail = "\n".join(
            output_lines[-100:]
        )

        raise RuntimeError(
            f"{stage} failed "
            f"with exit code "
            f"{return_code}.\n\n"
            f"{tail}"
        )


# ============================================================
# NORMALIZE INFERENCE RESULT
# ============================================================

def normalize_result(
    raw: Any,
    video_path: Path,
) -> dict[str, Any]:

    # --------------------------------------------------------
    # Handle list-style results
    # --------------------------------------------------------

    if isinstance(
        raw,
        list,
    ):

        raw = {
            "windows": raw
        }

    if not isinstance(
        raw,
        dict,
    ):

        raise ValueError(
            "Inference result JSON must "
            "contain an object or list."
        )

    # --------------------------------------------------------
    # Locate temporal results
    # --------------------------------------------------------

    windows_raw = (
        raw.get("windows")
        or raw.get("window_results")
        or raw.get("results")
        or []
    )

    windows: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # Normalize every temporal window
    # --------------------------------------------------------

    for index, item in enumerate(
        windows_raw,
        start=1,
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue

        classification = str(
            item.get(
                "classification"
            )
            or item.get(
                "prediction"
            )
            or "Normal"
        ).strip()

        # ----------------------------------------------------
        # Normalize classification
        # ----------------------------------------------------

        if classification not in CLASSES:

            lower = (
                classification.lower()
            )

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

            elif "fire" in lower:

                classification = (
                    "Fire"
                )

            elif "fight" in lower:

                classification = (
                    "Fight"
                )

            else:

                classification = (
                    "Normal"
                )

        windows.append(
            {
                "window": item.get(
                    "window",
                    index,
                ),

                "start": float(
                    item.get(
                        "start",
                        0.0,
                    )
                ),

                "end": float(
                    item.get(
                        "end",
                        0.0,
                    )
                ),

                "classification": (
                    classification
                ),

                "evidence": str(
                    item.get(
                        "evidence",
                        "",
                    )
                ),

                "incident_summary": str(
                    item.get(
                        "incident_summary",
                    )
                    or item.get(
                        "summary",
                    )
                    or ""
                ),

                "processing_time": (
                    item.get(
                        "processing_time"
                    )
                ),

                "error": item.get(
                    "error"
                ),
            }
        )

    # ========================================================
    # FINAL CLASSIFICATION
    # ========================================================

    classification = (
        raw.get(
            "classification"
        )
        or raw.get(
            "final_classification"
        )
        or raw.get(
            "prediction"
        )
    )

    if classification not in CLASSES:

        counts = {
            label: 0
            for label in CLASSES
        }

        for window in windows:

            label = window[
                "classification"
            ]

            if label in counts:

                counts[
                    label
                ] += 1

        if windows:

            classification = max(
                counts,
                key=counts.get,
            )

        else:

            classification = (
                "Normal"
            )

    # ========================================================
    # INCIDENT DATA
    # ========================================================

    incident_raw = (
        raw.get(
            "incident"
        )
        or {}
    )

    if not isinstance(
        incident_raw,
        dict,
    ):

        incident_raw = {}

    threat = (
        incident_raw.get(
            "threat_level"
        )
        or raw.get(
            "threat_level"
        )
    )

    summary = (
        incident_raw.get(
            "summary"
        )
        or raw.get(
            "summary"
        )
    )

    recommended_action = (
        incident_raw.get(
            "recommended_action"
        )
        or raw.get(
            "recommended_action"
        )
    )

    # ========================================================
    # DEFAULT THREAT LEVEL
    # ========================================================

    if not threat:

        threat = {
            "Normal": "LOW",
            "Fire": "CRITICAL",
            "Fight": "HIGH",
            "Road Accident": "HIGH",
        }.get(
            classification,
            "LOW",
        )

    # ========================================================
    # DEFAULT SUMMARY
    # ========================================================

    if not summary:

        if classification == "Normal":

            summary = (
                "No visible evidence of "
                "fire, fight, or road accident "
                "was detected."
            )

        else:

            summary = (
                f"{classification} detected "
                "in the analyzed video."
            )

    # ========================================================
    # DEFAULT RECOMMENDED ACTION
    # ========================================================

    if not recommended_action:

        recommended_action = {

            "Normal": (
                "No immediate action required."
            ),

            "Fire": (
                "Immediately alert emergency "
                "and fire-response personnel."
            ),

            "Fight": (
                "Security personnel should "
                "investigate the incident "
                "immediately."
            ),

            "Road Accident": (
                "Notify the appropriate "
                "emergency or traffic-response "
                "personnel."
            ),
        }[
            classification
        ]

    # ========================================================
    # API RESULT
    # ========================================================

    return {

        "analysis_id": None,

        "status": "completed",

        "video": {
            "filename": video_path.name,
            "path": str(video_path),
        },

        "model": raw.get(
            "model",
            "Qwen2.5-VL-7B-Instruct + LoRA",
        ),

        "final_classification": (
            classification
        ),

        "threat_level": threat,

        "summary": summary,

        "recommended_action": (
            recommended_action
        ),

        "incident": {

            "start": incident_raw.get(
                "start"
            ),

            "end": incident_raw.get(
                "end"
            ),

            "threat_level": threat,

            "summary": summary,

            "recommended_action": (
                recommended_action
            ),
        },

        "windows": windows,

        "window_counts": raw.get(
            "window_counts"
        ),

        "episodes": raw.get(
            "episodes",
            [],
        ),

        "processing_time": raw.get(
            "processing_time"
        ),

        # Preserve original output.
        "raw_result": raw,
    }


# ============================================================
# SAVE RESULT
# ============================================================

def save_job_result(
    job_id: str,
    result: dict[str, Any],
) -> None:

    result_path = (
        JOB_ROOT
        / job_id
        / "result.json"
    )

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# ANALYSIS WORKER
# ============================================================

def process_analysis(
    job_id: str,
    input_video: Path,
) -> None:

    job_dir = (
        JOB_ROOT
        / job_id
    )

    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        # ====================================================
        # GPU LOCK
        # ====================================================

        with GPU_LOCK:

            # ================================================
            # STAGE 1
            # PREPROCESSING
            # ================================================

            set_job(
                job_id,
                status="preprocessing",
                stage="Preprocessing",
                progress=10,
                message=(
                    "Normalizing video "
                    "for Qwen inference."
                ),
            )

            preprocess_command = [

                str(
                    PYTHON_EXE
                ),

                str(
                    PREPROCESSOR
                ),

                str(
                    input_video
                ),
            ]

            run_process(
                preprocess_command,
                job_id,
                "Preprocessing",
            )

            # ================================================
            # FIND PREPROCESSED DIRECTORY
            # ================================================

            preprocessed_dir = (
                PROJECT_ROOT
                / "dataset"
                / "preprocessed"
                / input_video.stem
            )

            manifest = (
                find_manifest(
                    preprocessed_dir
                )
            )

            # ================================================
            # STAGE 2
            # QWEN INFERENCE
            # ================================================

            set_job(
                job_id,
                status="inference",
                stage=(
                    "Qwen2.5-VL Inference"
                ),
                progress=40,
                message=(
                    "Running SentinelAI "
                    "Qwen2.5-VL inference."
                ),
                manifest=str(
                    manifest
                ),
            )

            inference_command = [

                str(
                    PYTHON_EXE
                ),

                str(
                    INFERENCE_ENGINE
                ),

                str(
                    manifest
                ),
            ]

            run_process(
                inference_command,
                job_id,
                "Qwen2.5-VL Inference",
            )

            # ================================================
            # FIND RESULT
            # ================================================

            result_file = (
                find_inference_results(
                    preprocessed_dir
                )
            )

        # ====================================================
        # GPU LOCK RELEASED HERE
        #
        # Qwen process has exited, so its GPU memory is
        # released before result aggregation.
        # ====================================================

        # ====================================================
        # STAGE 3
        # AGGREGATION
        # ====================================================

        set_job(
            job_id,
            status="aggregation",
            stage="Aggregation",
            progress=85,
            message=(
                "Formatting temporal "
                "inference results."
            ),
        )

        raw_result = load_json(
            result_file
        )

        result = normalize_result(
            raw_result,
            input_video,
        )

        result[
            "analysis_id"
        ] = job_id

        # ====================================================
        # INCIDENT CREATION
        # ====================================================

        incident_id = None

        if (
            result[
                "final_classification"
            ]
            != "Normal"
        ):

            incident_id = (
                f"INC-"
                f"{len(INCIDENTS) + 1:04d}"
            )

            incident = {

                "id": incident_id,

                "incident_type": (
                    result[
                        "final_classification"
                    ]
                ),

                "threat_level": (
                    result[
                        "threat_level"
                    ]
                ),

                "location": (
                    "Uploaded Video"
                ),

                "camera_id": None,

                "date_time": utc_now(),

                "status": "NEW",

                "summary": (
                    result[
                        "summary"
                    ]
                ),

                "recommended_action": (
                    result[
                        "recommended_action"
                    ]
                ),

                "analysis_id": job_id,
            }

            INCIDENTS[
                incident_id
            ] = incident

        result[
            "incident_id"
        ] = incident_id

        # ====================================================
        # SAVE RESULT
        # ====================================================

        save_job_result(
            job_id,
            result,
        )

        # ====================================================
        # COMPLETE
        # ====================================================

        set_job(
            job_id,
            status="completed",
            stage="Complete",
            progress=100,
            message=(
                "Analysis completed."
            ),
            result=result,
            result_file=str(
                result_file
            ),
            completed_at=utc_now(),
        )

    except Exception as exc:

        # ----------------------------------------------------
        # Never let the background thread silently die.
        # Store the complete error in the job.
        # ----------------------------------------------------

        set_job(
            job_id,
            status="failed",
            stage="Error",
            progress=100,
            message=str(exc),
            error=repr(exc),
            completed_at=utc_now(),
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root() -> dict[str, Any]:

    return {

        "project": "SentinelAI",

        "service": "FastAPI backend",

        "status": "running",

        "docs": "/docs",
    }


# ============================================================
# SYSTEM STATUS
# ============================================================

@app.get(
    "/api/system/status"
)
def system_status() -> dict[str, Any]:

    active = any(

        job.get(
            "status"
        )
        in {
            "queued",
            "preprocessing",
            "inference",
            "aggregation",
        }

        for job in JOBS.values()
    )

    return {

        "backend": "connected",

        "ai_engine": (
            "busy"
            if active
            else "ready"
        ),

        "gpu": gpu_status(),

        "model": (
            "Qwen2.5-VL-7B-Instruct + LoRA"
        ),

        "live_cameras": 0,

        "active_analysis": active,

        "timestamp": utc_now(),
    }


# ============================================================
# UPLOAD + START ANALYSIS
# ============================================================

@app.post(
    "/api/analyze-video",
    status_code=202,
)
async def analyze_video(
    file: UploadFile = File(...),
):

    filename = Path(
        file.filename or ""
    ).name

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    if not filename:

        raise HTTPException(
            status_code=400,
            detail=(
                "No filename was provided."
            ),
        )

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                f"Allowed formats: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # ========================================================
    # CREATE JOB
    # ========================================================

    job_id = (
        uuid.uuid4()
        .hex[:12]
    )

    upload_dir = (
        UPLOAD_ROOT
        / job_id
    )

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = (
        upload_dir
        / filename
    )

    # ========================================================
    # SAVE UPLOAD IN CHUNKS
    # ========================================================

    total_size = 0

    try:

        with input_path.open(
            "wb"
        ) as output_file:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(
                    chunk
                )

                if (
                    total_size
                    > MAX_UPLOAD_BYTES
                ):

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Video exceeds "
                            "the 500 MB limit."
                        ),
                    )

                output_file.write(
                    chunk
                )

    except HTTPException:

        if input_path.exists():
            input_path.unlink()

        raise

    finally:

        await file.close()

    # ========================================================
    # REGISTER JOB
    # ========================================================

    JOBS[job_id] = {

        "analysis_id": job_id,

        "status": "queued",

        "stage": "Queued",

        "progress": 0,

        "message": (
            "Video accepted "
            "and queued."
        ),

        "filename": filename,

        "size_bytes": total_size,

        "created_at": utc_now(),

        "result": None,

        "error": None,

        "last_log": None,
    }

    # ========================================================
    # START BACKGROUND THREAD
    # ========================================================

    worker = threading.Thread(

        target=process_analysis,

        args=(
            job_id,
            input_path,
        ),

        daemon=True,
    )

    worker.start()

    # ========================================================
    # RETURN IMMEDIATELY
    # ========================================================

    return {

        "analysis_id": job_id,

        "status": "queued",

        "message": (
            "Video accepted "
            "for analysis."
        ),
    }


# ============================================================
# ANALYSIS STATUS
# ============================================================

@app.get(
    "/api/analyze-video/{analysis_id}"
)
def analysis_status(
    analysis_id: str,
):

    job = JOBS.get(
        analysis_id
    )

    if job is None:

        result_path = (
            JOB_ROOT
            / analysis_id
            / "result.json"
        )

        if result_path.exists():

            result = load_json(
                result_path
            )

            return {

                "analysis_id": (
                    analysis_id
                ),

                "status": (
                    "completed"
                ),

                "progress": 100,

                "result": result,
            }

        raise HTTPException(
            status_code=404,
            detail=(
                "Analysis ID not found."
            ),
        )

    return job


# ============================================================
# INCIDENTS
# ============================================================

@app.get(
    "/api/incidents"
)
def list_incidents():

    return list(
        INCIDENTS.values()
    )[::-1]


@app.get(
    "/api/incidents/{incident_id}"
)
def get_incident(
    incident_id: str,
):

    incident = INCIDENTS.get(
        incident_id
    )

    if incident is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Incident not found."
            ),
        )

    return incident


@app.post(
    "/api/incidents/{incident_id}/acknowledge"
)
def acknowledge_incident(
    incident_id: str,
):

    incident = INCIDENTS.get(
        incident_id
    )

    if incident is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Incident not found."
            ),
        )

    incident[
        "status"
    ] = "ACKNOWLEDGED"

    return incident


@app.post(
    "/api/incidents/{incident_id}/resolve"
)
def resolve_incident(
    incident_id: str,
):

    incident = INCIDENTS.get(
        incident_id
    )

    if incident is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Incident not found."
            ),
        )

    incident[
        "status"
    ] = "RESOLVED"

    return incident


# ============================================================
# EVIDENCE
# ============================================================

@app.get(
    "/api/evidence"
)
def list_evidence():

    # Do NOT fabricate evidence URLs.
    # Your current inference pipeline does not yet expose
    # generated evidence clips through the API.

    return []


# ============================================================
# ANALYTICS
# ============================================================

@app.get(
    "/api/analytics"
)
def analytics():

    counts = {
        label: 0
        for label in CLASSES
    }

    for incident in (
        INCIDENTS.values()
    ):

        label = incident.get(
            "incident_type"
        )

        if label in counts:

            counts[
                label
            ] += 1

    return {

        "incident_counts": counts,

        "total_incidents": len(
            INCIDENTS
        ),
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host="0.0.0.0",

        port=8000,

        reload=False,
    )