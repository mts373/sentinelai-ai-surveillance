import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from sentinelai_previous import (
    MODEL_NAME,
    WINDOW_SECONDS,
    OVERLAP_SECONDS,
    NUM_FRAMES,
    DETECTION_PROMPT,
    get_video_info,
    sample_window,
    qwen_generate,
    parse_detection,
    classify_detection,
    build_episodes,
    classify_episode,
    generate_incident_report,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SentinelAI",
    page_icon="🚨",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .danger-box {
        padding: 20px;
        border-radius: 12px;
        background: #ffe5e5;
        border: 2px solid #ff4b4b;
    }

    .success-box {
        padding: 20px;
        border-radius: 12px;
        background: #e7f7ed;
        border: 2px solid #28a745;
    }

    .info-box {
        padding: 20px;
        border-radius: 12px;
        background: #eef4ff;
        border: 2px solid #4c8bf5;
    }

    .metric-label {
        font-size: 14px;
        color: #777;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚨 SentinelAI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered CCTV incident detection and analysis'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "model" not in st.session_state:
    st.session_state.model = None

if "processor" not in st.session_state:
    st.session_state.processor = None


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_qwen():

    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16
            if device == "cuda"
            else torch.float32,
            device_map="auto",
        )
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_NAME
        )
    )

    return model, processor


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ SentinelAI")

    mode = st.radio(
        "Input Mode",
        [
            "📁 Upload Video",
            "📷 Live Webcam",
        ],
    )

    st.divider()

    st.subheader("Detection Classes")

    st.write("🟢 Normal")
    st.write("🔥 Fire")
    st.write("👊 Fight")
    st.write("🚗 Road Accident")

    st.divider()

    st.caption(
        "Model: Qwen2.5-VL-7B-Instruct"
    )

    st.caption(
        "Temporal analysis: "
        "10s windows / 5s overlap"
    )

    if torch.cuda.is_available():

        st.success(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        st.warning(
            "CUDA unavailable. "
            "Inference will be slow."
        )


# ============================================================
# THREAT POLICY
# ============================================================

def get_threat_level(
    classification
):

    if classification == "Fire":
        return "HIGH"

    if classification == "Road Accident":
        return "HIGH"

    if classification == "Fight":
        return "MEDIUM"

    return "LOW"


def get_action(
    classification
):

    if classification == "Fire":

        return (
            "Immediately alert security personnel "
            "and contact emergency/fire services."
        )

    if classification == "Road Accident":

        return (
            "Alert security personnel and "
            "contact emergency medical services."
        )

    if classification == "Fight":

        return (
            "Alert security personnel and "
            "monitor the situation for escalation."
        )

    return (
        "No immediate action required."
    )


# ============================================================
# RESULT DISPLAY
# ============================================================

def display_result(
    result
):

    classification = result[
        "classification"
    ]

    threat = result[
        "threat_level"
    ]

    if classification == "Normal":

        st.success(
            "🟢 NORMAL — No incident detected"
        )

    elif classification == "Fire":

        st.error(
            "🔥 FIRE DETECTED"
        )

    elif classification == "Fight":

        st.warning(
            "👊 FIGHT DETECTED"
        )

    elif classification == "Road Accident":

        st.error(
            "🚗 ROAD ACCIDENT DETECTED"
        )

    else:

        st.warning(
            classification
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Classification",
            classification
        )

    with col2:

        st.metric(
            "Threat Level",
            threat
        )

    with col3:

        if result.get(
            "incident_start_seconds"
        ) is not None:

            start = result[
                "incident_start_seconds"
            ]

            end = result[
                "incident_end_seconds"
            ]

            st.metric(
                "Incident Time",
                f"{start:.1f}s – {end:.1f}s"
            )

        else:

            st.metric(
                "Incident Time",
                "None"
            )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "📝 Incident Summary"
        )

        st.write(
            result.get(
                "incident_summary",
                ""
            )
        )

    with col2:

        st.subheader(
            "🛡️ Recommended Action"
        )

        st.write(
            result.get(
                "recommended_action",
                ""
            )
        )

    if result.get("evidence"):

        st.subheader(
            "🔎 Evidence"
        )

        for evidence in result[
            "evidence"
        ]:

            st.info(
                evidence
            )

    if result.get("observations"):

        st.subheader(
            "Temporal Observations"
        )

        st.write(
            " → ".join(
                result[
                    "observations"
                ]
            )
        )


# ============================================================
# UPLOAD MODE
# ============================================================

if mode == "📁 Upload Video":

    st.header(
        "📁 Analyze CCTV Video"
    )

    uploaded_file = st.file_uploader(
        "Upload a CCTV video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
        ],
    )

    if uploaded_file:

        suffix = Path(
            uploaded_file.name
        ).suffix

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        )

        temp_file.write(
            uploaded_file.read()
        )

        temp_file.close()

        video_path = Path(
            temp_file.name
        )

        st.video(
            str(video_path)
        )

        if st.button(
            "🚨 Analyze Video",
            type="primary",
            use_container_width=True,
        ):

            model, processor = load_qwen()

            fps, total_frames, duration = (
                get_video_info(
                    video_path
                )
            )

            st.info(
                f"Video duration: "
                f"{duration:.1f}s | "
                f"FPS: {fps:.1f}"
            )

            step = (
                WINDOW_SECONDS
                - OVERLAP_SECONDS
            )

            windows = []

            start = 0.0

            while start < duration:

                end = min(
                    start + WINDOW_SECONDS,
                    duration
                )

                windows.append(
                    (start, end)
                )

                if end >= duration:
                    break

                start += step

            progress = st.progress(
                0
            )

            status = st.empty()

            all_windows = []

            pipeline_start = time.time()

            for index, (
                start,
                end
            ) in enumerate(
                windows,
                start=1
            ):

                status.write(
                    f"Analyzing window "
                    f"{index}/{len(windows)}: "
                    f"{start:.1f}s → {end:.1f}s"
                )

                frames = sample_window(
                    video_path,
                    start,
                    end
                )

                raw_output = qwen_generate(
                    model,
                    processor,
                    frames,
                    DETECTION_PROMPT,
                    300
                )

                detection = parse_detection(
                    raw_output
                )

                classification = (
                    classify_detection(
                        detection
                    )
                )

                all_windows.append(
                    {
                        "window_index":
                            index,

                        "start":
                            start,

                        "end":
                            end,

                        "classification":
                            classification,

                        "detection":
                            detection,

                        "raw_output":
                            raw_output,
                    }
                )

                progress.progress(
                    index / len(windows)
                )

            # ------------------------------------------------
            # TEMPORAL EPISODES
            # ------------------------------------------------

            incident_windows = [
                x
                for x in all_windows
                if x["classification"]
                in {
                    "Fire",
                    "Fight",
                    "Road Accident",
                    "Multiple",
                }
            ]

            episodes = build_episodes(
                incident_windows
            )

            episode_results = []

            for episode in episodes:

                episode_results.append(
                    classify_episode(
                        episode
                    )
                )

            if episode_results:

                primary_episode = (
                    episode_results[0]
                )

                final_classification = (
                    primary_episode[
                        "primary_classification"
                    ]
                )

                report = (
                    generate_incident_report(
                        model,
                        processor,
                        primary_episode
                    )
                )

            else:

                primary_episode = None

                final_classification = (
                    "Normal"
                )

                report = {
                    "incident_summary":
                        "No fire, fight, or "
                        "road accident was detected.",

                    "threat_level":
                        "LOW",

                    "recommended_action":
                        "No immediate action required.",
                }

            # ------------------------------------------------
            # DETERMINISTIC SAFETY POLICY
            # ------------------------------------------------

            threat = get_threat_level(
                final_classification
            )

            action = get_action(
                final_classification
            )

            result = {

                "classification":
                    final_classification,

                "incident_start_seconds":
                    (
                        primary_episode[
                            "start_seconds"
                        ]
                        if primary_episode
                        else None
                    ),

                "incident_end_seconds":
                    (
                        primary_episode[
                            "end_seconds"
                        ]
                        if primary_episode
                        else None
                    ),

                "observations":
                    (
                        primary_episode[
                            "observations"
                        ]
                        if primary_episode
                        else []
                    ),

                "evidence":
                    (
                        primary_episode[
                            "evidence"
                        ]
                        if primary_episode
                        else []
                    ),

                "incident_summary":
                    report.get(
                        "incident_summary",
                        ""
                    ),

                "threat_level":
                    threat,

                "recommended_action":
                    action,

                "windows_analyzed":
                    len(all_windows),

                "processing_time_seconds":
                    time.time()
                    - pipeline_start,
            }

            st.session_state[
                "last_result"
            ] = result

            status.success(
                "✅ Analysis complete."
            )

            display_result(
                result
            )


# ============================================================
# LIVE WEBCAM MODE
# ============================================================

else:

    st.header(
        "📷 Live CCTV Monitor"
    )

    st.info(
        "For the hackathon demo, place your phone "
        "in front of the laptop webcam and play "
        "a CCTV/anomaly video on the phone."
    )

    st.warning(
        "Live analysis uses temporal windows. "
        "It is near-real-time, not frame-by-frame."
    )

    duration_limit = st.slider(
        "Demo duration (seconds)",
        min_value=30,
        max_value=300,
        value=60,
        step=10,
    )

    start_live = st.button(
        "▶️ START LIVE DETECTION",
        type="primary",
        use_container_width=True,
    )

    if start_live:

        model, processor = load_qwen()

        cap = cv2.VideoCapture(
            0
        )

        if not cap.isOpened():

            st.error(
                "Could not open webcam. "
                "Check that your camera is connected "
                "and not being used by another application."
            )

            st.stop()

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        video_placeholder = st.empty()

        status_placeholder = st.empty()

        result_placeholder = st.empty()

        timeline_placeholder = st.empty()

        # ----------------------------------------------------
        # LIVE BUFFER
        # ----------------------------------------------------

        buffer = []

        buffer_start_time = time.time()

        live_start_time = time.time()

        timeline = []

        last_classification = "Normal"

        try:

            while (
                time.time()
                - live_start_time
                < duration_limit
            ):

                success, frame = (
                    cap.read()
                )

                if not success:

                    st.error(
                        "Failed to read "
                        "from webcam."
                    )

                    break

                # OpenCV BGR → RGB
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                video_placeholder.image(
                    rgb,
                    channels="RGB",
                    use_container_width=True,
                )

                buffer.append(
                    rgb
                )

                elapsed_buffer = (
                    time.time()
                    - buffer_start_time
                )

                # ------------------------------------------------
                # ANALYZE EVERY 10 SECONDS
                # ------------------------------------------------

                if elapsed_buffer >= 10:

                    status_placeholder.info(
                        "🔎 Analyzing latest "
                        "10-second CCTV window..."
                    )

                    # Select 16 evenly distributed frames
                    indices = np.linspace(
                        0,
                        len(buffer) - 1,
                        NUM_FRAMES,
                        dtype=int,
                    )

                    selected_frames = [
                        buffer[i]
                        for i in indices
                    ]

                    pil_frames = [
                        __import__(
                            "PIL.Image",
                            fromlist=["Image"]
                        ).Image.fromarray(
                            frame
                        )
                        for frame in selected_frames
                    ]

                    raw_output = qwen_generate(
                        model,
                        processor,
                        pil_frames,
                        DETECTION_PROMPT,
                        300
                    )

                    detection = (
                        parse_detection(
                            raw_output
                        )
                    )

                    classification = (
                        classify_detection(
                            detection
                        )
                    )

                    last_classification = (
                        classification
                    )

                    timestamp = (
                        time.time()
                        - live_start_time
                    )

                    timeline.append(
                        {
                            "time":
                                timestamp,

                            "classification":
                                classification,

                            "detection":
                                detection,
                        }
                    )

                    # ------------------------------------------------
                    # DISPLAY CURRENT DETECTION
                    # ------------------------------------------------

                    if classification == "Fire":

                        result_placeholder.error(
                            "🔥 FIRE DETECTED — HIGH THREAT"
                        )

                    elif classification == "Road Accident":

                        result_placeholder.error(
                            "🚗 ROAD ACCIDENT DETECTED — HIGH THREAT"
                        )

                    elif classification == "Fight":

                        result_placeholder.warning(
                            "👊 FIGHT DETECTED — MEDIUM THREAT"
                        )

                    elif classification == "Normal":

                        result_placeholder.success(
                            "🟢 NORMAL — No incident detected"
                        )

                    else:

                        result_placeholder.warning(
                            f"⚠️ {classification}"
                        )

                    # ------------------------------------------------
                    # TIMELINE
                    # ------------------------------------------------

                    timeline_text = (
                        "### Incident Timeline\n\n"
                    )

                    for item in timeline:

                        cls = (
                            item[
                                "classification"
                            ]
                        )

                        if cls == "Fire":

                            icon = "🔥"

                        elif cls == "Fight":

                            icon = "👊"

                        elif cls == "Road Accident":

                            icon = "🚗"

                        else:

                            icon = "🟢"

                        timeline_text += (
                            f"- `{item['time']:.1f}s` "
                            f"{icon} {cls}\n"
                        )

                    timeline_placeholder.markdown(
                        timeline_text
                    )

                    # ------------------------------------------------
                    # RESET BUFFER
                    # ------------------------------------------------

                    buffer = []

                    buffer_start_time = (
                        time.time()
                    )

                    status_placeholder.success(
                        "✅ Window analyzed. "
                        "Collecting next window..."
                    )

                time.sleep(
                    0.01
                )

        finally:

            cap.release()

            cv2.destroyAllWindows()

        status_placeholder.success(
            "🛑 Live detection finished."
        )

        st.write(
            f"Last classification: "
            f"**{last_classification}**"
        )