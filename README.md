# Vision Language Model Based Incident Intelligence Platform for Smart Surveillance and Emergency Response

**SentinelAI** is an AI-powered incident intelligence platform that transforms traditional CCTV footage into structured surveillance intelligence using Vision-Language Models.

The current MVP analyzes uploaded surveillance videos using **Qwen2.5-VL**, performs temporal window-based analysis, classifies selected incidents, generates incident summaries and threat levels, extracts visual evidence, creates structured incidents, supports human review, routes emergency responses, and presents the results through a SOC-style web dashboard.

---

## 1. Project Status

**Current MVP: Working**

The implemented prototype currently supports:

- CCTV/video upload
- Memory-controlled video preprocessing
- Temporal video windowing
- Qwen2.5-VL visual analysis
- LoRA/QLoRA-based inference
- Window-level classification
- Video-level temporal aggregation
- Incident generation
- Threat-level estimation
- AI-generated incident summaries
- Visual evidence frame extraction
- Evidence API
- Synchronized video playback
- Temporal analysis timeline
- Evidence gallery
- Incident management
- Human review and correction records
- Emergency response routing
- Email notification integration through Resend
- FastAPI backend
- SOC-style web dashboard
- MLflow experiment tracking
- Held-out evaluation

### Current AI Classes

- **Normal**
- **Fire**
- **Fight**
- **Road Accident**

> Unauthorized Entry, suspicious activity, crowd panic, and other incident categories are planned future extensions and are not currently completed model classes.

---

# 2. Problem

Traditional CCTV systems are primarily passive recording systems. They continuously store video but depend heavily on human operators to monitor multiple camera feeds.

In practical environments, operators cannot continuously watch dozens or hundreds of cameras. Important incidents such as fires, road accidents, physical fights, unauthorized intrusions, and suspicious activities can therefore be missed or detected too late.

Delayed detection can increase:

- Emergency response time
- Property damage
- Public safety risks
- Operational costs
- Dependence on continuous human monitoring

Most conventional surveillance systems focus on recording, playback, or basic object detection rather than understanding complex real-world events.

---

# 3. Proposed Solution

SentinelAI explores an AI-assisted surveillance approach in which Vision-Language Models analyze surveillance video and transform visual observations into structured incident intelligence.

The target platform is designed to:

- Analyze surveillance video
- Understand ongoing activities
- Classify selected incidents
- Estimate threat severity
- Generate incident summaries
- Identify relevant temporal sections
- Generate visual evidence
- Display incidents through a centralized dashboard
- Support human verification
- Route incidents to appropriate response departments
- Provide notification capabilities

The current MVP focuses on uploaded-video analysis while providing an architecture that can be extended to live CCTV streams.

---

# 4. System Architecture

```text
CCTV / Uploaded Video
        │
        ▼
Video Preprocessing
        │
        ├── Resolution normalization
        ├── FPS normalization
        └── Memory-controlled processing
        │
        ▼
Temporal Windowing
        │
        └── Approximately 10-second windows
        │
        ▼
Qwen2.5-VL + LoRA
        │
        ▼
Window-Level Classification
        │
        ├── Normal
        ├── Fire
        ├── Fight
        └── Road Accident
        │
        ▼
Temporal / Video-Level Aggregation
        │
        ▼
Incident Intelligence
        │
        ├── Classification
        ├── Threat Level
        ├── Summary
        ├── Recommended Action
        └── Temporal Episodes
        │
        ▼
Visual Evidence Extraction
        │
        ▼
FastAPI Backend
        │
        ├── Incident API
        ├── Evidence API
        ├── Human Review API
        ├── Analytics API
        └── Notification Integration
        │
        ▼
SentinelAI Web Dashboard