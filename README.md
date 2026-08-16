\# SentinelAI — AI-Powered Incident Intelligence



SentinelAI is an AI-powered surveillance prototype that transforms CCTV video into structured incident intelligence using a Vision-Language Model.



The system processes uploaded surveillance video, performs memory-controlled video preprocessing and temporal analysis, classifies the observed activity, aggregates window-level predictions, and exposes the analysis through a FastAPI backend connected to a web dashboard.



\---



\## Project Status



\*\*Current MVP:\*\* Working



The current prototype supports:



\- CCTV video upload

\- Automatic video preprocessing

\- Temporal video windowing

\- Qwen2.5-VL based visual analysis

\- LoRA/QLoRA fine-tuned inference

\- Four classification categories

\- Window-level inference

\- Video-level result aggregation

\- FastAPI backend

\- Web dashboard integration

\- MLflow experiment tracking

\- Fresh held-out evaluation



\### Supported AI classes



\- Normal

\- Fire

\- Fight

\- Road Accident



\*\*Unauthorized Entry\*\* is part of the planned future scope and is not currently represented as a completed model class.



\---



\# 1. Problem



Traditional CCTV systems primarily record video and depend on human operators to continuously monitor multiple camera feeds.



This creates a practical limitation: important incidents can be missed when operators are monitoring many cameras simultaneously.



SentinelAI explores an AI-assisted approach in which surveillance video is analyzed automatically to identify selected incidents and provide structured information to the operator.



\---



\# 2. Proposed Solution



SentinelAI uses a Vision-Language Model to analyze surveillance video instead of relying only on conventional object detection.



The current pipeline is:



```text

CCTV / Uploaded Video

&#x20;       │

&#x20;       ▼

Video Preprocessing

&#x20;       │

&#x20;       ├── Resolution normalization

&#x20;       ├── FPS normalization

&#x20;       └── Memory-controlled processing

&#x20;       │

&#x20;       ▼

Temporal Windowing

&#x20;       │

&#x20;       └── 10-second analysis windows

&#x20;       │

&#x20;       ▼

Qwen2.5-VL + LoRA

&#x20;       │

&#x20;       ▼

Window-Level Classification

&#x20;       │

&#x20;       ├── Normal

&#x20;       ├── Fire

&#x20;       ├── Fight

&#x20;       └── Road Accident

&#x20;       │

&#x20;       ▼

Temporal / Video-Level Aggregation

&#x20;       │

&#x20;       ▼

FastAPI Backend

&#x20;       │

&#x20;       ▼

SentinelAI Web Dashboard

# 13. Web Dashboard

SentinelAI uses a web dashboard as the user-facing interface for the MVP.

The dashboard communicates with the FastAPI backend to submit videos and retrieve analysis results.

The current backend integration follows:

```text
Web Dashboard
      │
      │ POST /api/analyze-video
      ▼
FastAPI Backend
      │
      ▼
Video Upload
      │
      ▼
video_preprocessor.py
      │
      ▼
Temporal Manifest
      │
      ▼
inference_engine.py
      │
      ▼
inference_results.json
      │
      ▼
FastAPI
      │
      ▼
Web Dashboard

