# ============================================================
# SENTINELAI - STREAMLIT APPLICATION
# ============================================================
#
# File:
# C:\SentinelAI_Qwen\app.py
#
# Pipeline:
#
# Upload
#   ↓
# Video Preprocessing
#   ↓
# 10-second temporal windows
#   ↓
# Qwen2.5-VL + SentinelAI LoRA
#   ↓
# Window predictions
#   ↓
# Temporal timeline
#   ↓
# Video-level decision
#
# ============================================================

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATHS
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
# IMPORT EXISTING WORKING PIPELINE
# ============================================================

from video_preprocessor import (
    VideoPreprocessor,
)

from inference_engine import (
    process_manifest,
)


# ============================================================
# CONFIGURATION
# ============================================================

PREPROCESSED_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "preprocessed"
)

APP_RESULTS_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "app_results"
)

WINDOW_SECONDS = 10.0
TARGET_FPS = 2.0
MAX_FRAMES = 20
MAX_WIDTH = 1280
MAX_HEIGHT = 720

CLASSES = [
    "Normal",
    "Fire",
    "Fight",
    "Road Accident",
]


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="SentinelAI",
    page_icon="🚨",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.7;
        margin-bottom: 25px;
    }

    .timeline-normal {
        padding: 12px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 5px solid #2e7d32;
        background: rgba(46,125,50,0.10);
    }

    .timeline-anomaly {
        padding: 12px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 5px solid #d32f2f;
        background: rgba(211,47,47,0.10);
    }

    .timeline-error {
        padding: 12px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 5px solid #f57c00;
        background: rgba(245,124,0,0.10);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">SentinelAI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered CCTV anomaly detection'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Pipeline"
    )

    st.write(
        "1. Upload video"
    )

    st.write(
        "2. Normalize video"
    )

    st.write(
        "3. Create temporal windows"
    )

    st.write(
        "4. Qwen2.5-VL + LoRA"
    )

    st.write(
        "5. Analyze windows"
    )

    st.write(
        "6. Generate video-level result"
    )

    st.divider()

    st.caption(
        "Detection Classes"
    )

    for label in CLASSES:

        st.write(
            f"• {label}"
        )

    st.divider()

    st.caption(
        "Inference Settings"
    )

    st.write(
        f"Window: {WINDOW_SECONDS:.0f}s"
    )

    st.write(
        f"Sampling: {TARGET_FPS:.1f} FPS"
    )

    st.write(
        f"Maximum frames/window: {MAX_FRAMES}"
    )

    st.write(
        f"Maximum resolution: "
        f"{MAX_WIDTH} × {MAX_HEIGHT}"
    )


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload CCTV / surveillance video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
        "webm",
    ],
)


# ============================================================
# HELPERS
# ============================================================

def safe_filename(
    filename: str,
) -> str:

    stem = Path(
        filename
    ).stem

    suffix = Path(
        filename
    ).suffix.lower()

    stem = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        stem,
    )

    if not stem:

        stem = "uploaded_video"

    if not suffix:

        suffix = ".mp4"

    return (
        stem + suffix
    )


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_predictions(
    results,
):
    """
    Convert window-level predictions into one
    video-level prediction.

    IMPORTANT:

    This is an application aggregation rule.
    It is NOT a calibrated probability.

    We select the anomaly class with the highest
    number of positive windows.

    If no anomaly is detected:
        Normal
    """

    counts = {
        "Normal": 0,
        "Fire": 0,
        "Fight": 0,
        "Road Accident": 0,
    }

    valid_results = []

    for result in results:

        label = result.get(
            "classification"
        )

        if label in CLASSES:

            counts[label] += 1

            valid_results.append(
                result
            )

    if not valid_results:

        return {
            "classification": "Unknown",
            "agreement": 0.0,
            "reason": (
                "No valid model predictions "
                "were produced."
            ),
            "window_counts": counts,
            "analyzed_windows": 0,
        }

    anomaly_classes = [
        "Fire",
        "Fight",
        "Road Accident",
    ]

    strongest = max(
        anomaly_classes,
        key=lambda label: counts[label],
    )

    strongest_count = counts[
        strongest
    ]

    if strongest_count > 0:

        final_label = strongest

    else:

        final_label = "Normal"

    total = len(
        valid_results
    )

    agreement = (
        counts[final_label]
        / total
    )

    # --------------------------------------------------------
    # Selected anomaly windows
    # --------------------------------------------------------

    selected = [
        result
        for result in valid_results
        if result.get(
            "classification"
        ) == final_label
    ]

    evidence = []
    summaries = []

    for result in selected:

        value = str(
            result.get(
                "evidence",
                "",
            )
        ).strip()

        if value:

            evidence.append(
                value
            )

        value = str(
            result.get(
                "incident_summary",
                "",
            )
        ).strip()

        if value:

            summaries.append(
                value
            )

    evidence = list(
        dict.fromkeys(
            evidence
        )
    )

    summaries = list(
        dict.fromkeys(
            summaries
        )
    )

    if final_label == "Normal":

        reason = (
            "No Fire, Fight, or Road Accident "
            "was detected in the analyzed windows."
        )

    else:

        reason = (
            f"{strongest_count} of {total} "
            f"analyzed windows were classified "
            f"as {final_label}."
        )

    return {
        "classification":
            final_label,

        "agreement":
            agreement,

        "reason":
            reason,

        "evidence":
            " ".join(
                evidence[:3]
            ),

        "incident_summary":
            " ".join(
                summaries[:3]
            ),

        "window_counts":
            counts,

        "analyzed_windows":
            total,
    }


# ============================================================
# SAVE UPLOAD
# ============================================================

def save_uploaded_video(
    uploaded_file,
):

    APP_RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_directory = Path(
        tempfile.mkdtemp(
            prefix="sentinelai_",
            dir=str(
                APP_RESULTS_ROOT
            ),
        )
    )

    input_path = (
        temp_directory
        / safe_filename(
            uploaded_file.name
        )
    )

    with open(
        input_path,
        "wb",
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    return (
        temp_directory,
        input_path,
    )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_pipeline(
    uploaded_file,
    progress_callback,
):

    temp_directory, input_path = (
        save_uploaded_video(
            uploaded_file
        )
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    progress_callback(
        0.05,
        "Saving uploaded video..."
    )

    progress_callback(
        0.10,
        "Inspecting and preprocessing video..."
    )

    video_processor = VideoPreprocessor(
        output_root=PREPROCESSED_ROOT,
        window_seconds=WINDOW_SECONDS,
        target_fps=TARGET_FPS,
        max_frames_per_window=MAX_FRAMES,
        max_width=MAX_WIDTH,
        max_height=MAX_HEIGHT,
    )

    preprocessing_result = (
        video_processor.process(
            input_path,
            clean=True,
        )
    )

    windows = (
        preprocessing_result.windows
    )

    if not windows:

        raise RuntimeError(
            "No temporal windows were produced."
        )

    output_directory = Path(
        preprocessing_result.output_directory
    )

    manifest_path = (
        output_directory
        / "manifest.json"
    )

    if not manifest_path.exists():

        raise RuntimeError(
            "manifest.json was not created."
        )

    progress_callback(
        0.30,
        f"Created {len(windows)} temporal windows."
    )

    # ========================================================
    # QWEN INFERENCE
    # ========================================================

    progress_callback(
        0.35,
        "Loading Qwen2.5-VL + SentinelAI LoRA..."
    )

    results = process_manifest(
        manifest_path
    )

    progress_callback(
        0.90,
        "Analyzing temporal predictions..."
    )

    # ========================================================
    # VIDEO LEVEL RESULT
    # ========================================================

    final_result = (
        aggregate_predictions(
            results
        )
    )

    progress_callback(
        0.95,
        "Saving results..."
    )

    complete_result = {

        "input_video":
            uploaded_file.name,

        "manifest":
            str(
                manifest_path
            ),

        "final_result":
            final_result,

        "window_results":
            results,
    }

    result_path = (
        temp_directory
        / "sentinelai_result.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            complete_result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    progress_callback(
        1.0,
        "Analysis complete."
    )

    return {
        "manifest":
            str(
                manifest_path
            ),

        "preprocessed_directory":
            str(
                output_directory
            ),

        "final_result":
            final_result,

        "window_results":
            results,

        "result_path":
            str(
                result_path
            ),
    }


# ============================================================
# TEMPORAL TIMELINE
# ============================================================

def display_timeline(
    results,
):

    st.subheader(
        "Temporal Detection Timeline"
    )

    if not results:

        st.warning(
            "No temporal results available."
        )

        return

    for index, result in enumerate(
        results,
        start=1,
    ):

        start = float(
            result.get(
                "start",
                0.0,
            )
        )

        end = float(
            result.get(
                "end",
                0.0,
            )
        )

        label = result.get(
            "classification"
        )

        if label in CLASSES:

            if label == "Normal":

                icon = "🟢"
                css_class = (
                    "timeline-normal"
                )

            else:

                icon = "🔴"
                css_class = (
                    "timeline-anomaly"
                )

        else:

            icon = "🟠"
            css_class = (
                "timeline-error"
            )

            label = "ERROR"

        st.markdown(
            f"""
            <div class="{css_class}">
                <b>{icon} Window {index}</b>
                &nbsp;&nbsp;
                <b>{start:.2f}s → {end:.2f}s</b>
                &nbsp;&nbsp;
                <b>{label}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# DISPLAY ANOMALY WINDOWS
# ============================================================

def display_anomaly_windows(
    results,
):

    anomaly_results = [
        result
        for result in results
        if result.get(
            "classification"
        ) in (
            "Fire",
            "Fight",
            "Road Accident",
        )
    ]

    if not anomaly_results:

        st.success(
            "No anomaly windows detected."
        )

        return

    st.subheader(
        "Detected Incident Windows"
    )

    for result in anomaly_results:

        start = float(
            result.get(
                "start",
                0,
            )
        )

        end = float(
            result.get(
                "end",
                0,
            )
        )

        label = result.get(
            "classification"
        )

        st.error(
            f"🚨 {label} — "
            f"{start:.2f}s → {end:.2f}s"
        )

        evidence = str(
            result.get(
                "evidence",
                "",
            )
        ).strip()

        summary = str(
            result.get(
                "incident_summary",
                "",
            )
        ).strip()

        if evidence:

            st.write(
                f"**Evidence:** {evidence}"
            )

        if summary:

            st.write(
                f"**Summary:** {summary}"
            )


# ============================================================
# MAIN UI
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a surveillance video to begin."
    )

    st.markdown(
        """
        ### SentinelAI

        The system will:

        **Upload → Preprocess → Temporal Windows →
        Qwen2.5-VL + LoRA → Detection → Timeline**

        The application currently detects:

        - Normal
        - Fire
        - Fight
        - Road Accident
        """

    )

else:

    # ========================================================
    # VIDEO PREVIEW
    # ========================================================

    st.video(
        uploaded_file
    )

    size_mb = (
        uploaded_file.size
        / 1024
        / 1024
    )

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.write(
            f"**File:** "
            f"{uploaded_file.name}"
        )

    with col2:

        st.write(
            f"**Size:** "
            f"{size_mb:.2f} MB"
        )

    st.divider()

    # ========================================================
    # ANALYZE
    # ========================================================

    analyze = st.button(
        "🚨 Analyze Video",
        type="primary",
        use_container_width=True,
    )

    if analyze:

        progress_bar = st.progress(
            0.0
        )

        status = st.empty()

        try:

            with st.spinner(
                "SentinelAI is analyzing the video..."
            ):

                result = run_pipeline(
                    uploaded_file,
                    lambda value, message: (
                        progress_bar.progress(
                            value
                        ),
                        status.info(
                            message
                        ),
                    ),
                )

            progress_bar.progress(
                1.0
            )

            status.success(
                "Analysis complete."
            )

            final = result[
                "final_result"
            ]

            classification = (
                final[
                    "classification"
                ]
            )

            agreement = (
                final[
                    "agreement"
                ]
            )

            results = result[
                "window_results"
            ]

            # =================================================
            # FINAL DETECTION
            # =================================================

            st.divider()

            st.header(
                "Final Detection"
            )

            if classification == "Normal":

                st.success(
                    f"## ✅ {classification}"
                )

            elif classification in (
                "Fire",
                "Fight",
                "Road Accident",
            ):

                st.error(
                    f"## 🚨 {classification}"
                )

            else:

                st.warning(
                    f"## ⚠️ {classification}"
                )

            col1, col2, col3 = st.columns(
                3
            )

            with col1:

                st.metric(
                    "Final Class",
                    classification,
                )

            with col2:

                st.metric(
                    "Analyzed Windows",
                    final[
                        "analyzed_windows"
                    ],
                )

            with col3:

                st.metric(
                    "Window Agreement",
                    f"{agreement * 100:.1f}%",
                )

            st.info(
                final[
                    "reason"
                ]
            )

            # =================================================
            # INCIDENT SUMMARY
            # =================================================

            if final.get(
                "incident_summary"
            ):

                st.subheader(
                    "Incident Summary"
                )

                st.write(
                    final[
                        "incident_summary"
                    ]
                )

            # =================================================
            # TIMELINE
            # =================================================

            st.divider()

            display_timeline(
                results
            )

            # =================================================
            # ANOMALY WINDOWS
            # =================================================

            st.divider()

            display_anomaly_windows(
                results
            )

            # =================================================
            # CLASS DISTRIBUTION
            # =================================================

            st.divider()

            st.subheader(
                "Prediction Distribution"
            )

            counts = final[
                "window_counts"
            ]

            chart_data = {
                "Class": CLASSES,
                "Windows": [
                    counts.get(
                        label,
                        0,
                    )
                    for label in CLASSES
                ],
            }

            st.bar_chart(
                chart_data,
                x="Class",
                y="Windows",
            )

            # =================================================
            # DETAILED WINDOWS
            # =================================================

            st.divider()

            st.subheader(
                "Detailed Window Results"
            )

            for index, result_item in enumerate(
                results,
                start=1,
            ):

                start = float(
                    result_item.get(
                        "start",
                        0,
                    )
                )

                end = float(
                    result_item.get(
                        "end",
                        0,
                    )
                )

                label = (
                    result_item.get(
                        "classification"
                    )
                    or "ERROR"
                )

                with st.expander(
                    f"Window {index}: "
                    f"{start:.2f}s → "
                    f"{end:.2f}s | "
                    f"{label}"
                ):

                    if result_item.get(
                        "evidence"
                    ):

                        st.write(
                            "**Evidence**"
                        )

                        st.write(
                            result_item[
                                "evidence"
                            ]
                        )

                    if result_item.get(
                        "incident_summary"
                    ):

                        st.write(
                            "**Incident Summary**"
                        )

                        st.write(
                            result_item[
                                "incident_summary"
                            ]
                        )

                    processing_time = (
                        result_item.get(
                            "processing_time"
                        )
                    )

                    if processing_time:

                        st.caption(
                            f"Processing time: "
                            f"{processing_time:.2f}s"
                        )

                    if result_item.get(
                        "error"
                    ):

                        st.error(
                            result_item[
                                "error"
                            ]
                        )

            # =================================================
            # DOWNLOAD JSON
            # =================================================

            st.divider()

            st.subheader(
                "Export"
            )

            result_json = json.dumps(
                {
                    "final_result":
                        final,

                    "window_results":
                        results,
                },
                indent=2,
                ensure_ascii=False,
            )

            st.download_button(
                label=(
                    "Download Analysis JSON"
                ),
                data=result_json,
                file_name=(
                    "sentinelai_analysis.json"
                ),
                mime=(
                    "application/json"
                ),
                use_container_width=True,
            )

            # =================================================
            # TECHNICAL DETAILS
            # =================================================

            with st.expander(
                "Technical Details"
            ):

                st.write(
                    "**Manifest:**"
                )

                st.code(
                    result[
                        "manifest"
                    ]
                )

                st.write(
                    "**Preprocessed directory:**"
                )

                st.code(
                    result[
                        "preprocessed_directory"
                    ]
                )

                st.write(
                    "**Saved result:**"
                )

                st.code(
                    result[
                        "result_path"
                    ]
                )

        except Exception as error:

            progress_bar.empty()

            status.empty()

            st.error(
                "SentinelAI analysis failed."
            )

            st.exception(
                error
            )

            st.warning(
                "Check the traceback before changing "
                "the model or inference engine."
            )