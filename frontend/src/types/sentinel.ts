/**
 * Types mirroring the real SentinelAI FastAPI backend (main.py).
 * Do not add fields the backend does not return.
 */

export type SentinelClass = "Normal" | "Fire" | "Fight" | "Road Accident";

export type ThreatLevel = "LOW" | "HIGH" | "CRITICAL" | string;

/** GET /api/system/status -> gpu (object, never a string) */
export interface GpuStatus {
  available: boolean;
  name: string | null;
  utilization_percent: number | null;
  memory_used_gb: number | null;
  memory_total_gb: number | null;
}

/** GET /api/system/status */
export interface SystemStatus {
  backend: string;
  ai_engine: string;
  gpu: GpuStatus;
  model: string;
  live_cameras: number;
  active_analysis: boolean;
  timestamp: string;
}

/** normalize_result() -> windows[] */
export interface HumanReview {
  verdict: ReviewVerdict;
  corrected_label: string | null;
  notes: string | null;
  reviewed_at: string;
  analysis_id: string;
  incident_id: string | null;
  ai_prediction: string;
}

export type ReviewVerdict = "correct" | "incorrect";


export interface AnalysisWindow {
  window: number;
  start: number;
  end: number;
  classification: SentinelClass | string;
  evidence: string;
  incident_summary: string;
  processing_time: number | null;
  error: string | null;
  /** Optional — only present when the backend reports it. Never fabricated. */
  confidence?: number | null;
  /** Optional evidence frame paths/URLs returned by the backend for this window. */
  frames?: string[] | null;
  evidence_frames?: string[] | null;
  /** Optional spatial localization, only when the model/backend provides it. */
  bbox?: [number, number, number, number] | null;
  regions?: Array<{ bbox?: [number, number, number, number] | null; label?: string | null }> | null;
}

export interface AnalysisIncidentBlock {
  start?: number | null;
  end?: number | null;
  threat_level?: ThreatLevel | null;
  summary?: string | null;
  recommended_action?: string | null;
  [key: string]: unknown;
}

/** normalize_result() */
export interface AnalysisResult {
  analysis_id: string | null;
  status: string;
  video: { filename: string; path: string; url?: string | null; duration?: number | null };
  video_url?: string | null;
  model: string;
  final_classification: SentinelClass | string;
  threat_level: ThreatLevel;
  summary: string;
  recommended_action: string;
  incident: AnalysisIncidentBlock;
  windows: AnalysisWindow[];
  window_counts?: Record<string, number> | null;
  episodes?: unknown;
  processing_time?: number | null;
  raw_result?: unknown;
  incident_id?: string | null;
  confidence?: number | null;
  evidence_frames?: Array<{
    url?: string | null;
    path?: string | null;
    timestamp?: number | null;
    window?: number | null;
    classification?: string | null;
    confidence?: number | null;
  }> | null;
  notification?: {
    status?: "SENT" | "FAILED" | "PENDING" | "UNKNOWN" | string;
    email_sent?: boolean | null;
    department?: string | null;
    recipient?: string | null;
    sent_at?: string | null;
    error?: string | null;
  } | null;
  human_review?: HumanReview | null;
}


export type AnalysisJobStatus =
  | "queued"
  | "preprocessing"
  | "inference"
  | "aggregation"
  | "completed"
  | "failed"
  | string;

/** GET /api/analyze-video/{analysis_id} */
export interface AnalysisJob {
  analysis_id: string;
  status: AnalysisJobStatus;
  stage?: string;
  progress: number;
  message?: string;
  filename?: string;
  size_bytes?: number;
  created_at?: string;
  result: AnalysisResult | null;
  error?: string | null;
  last_log?: string | null;
  manifest?: string;
  incident_id?: string | null;
}

/** POST /api/analyze-video (202) */
export interface AnalyzeVideoAccepted {
  analysis_id: string;
  status: string;
  message: string;
}

/** GET /api/incidents */
export interface Incident {
  id: string;
  incident_type: SentinelClass | string;
  threat_level: ThreatLevel;
  location: string;
  camera_id: string | null;
  date_time: string;
  status: string;
  summary: string;
  recommended_action: string;
  analysis_id: string;
  notification?: {
    status?: "SENT" | "FAILED" | "PENDING" | "UNKNOWN" | string;
    email_sent?: boolean | null;
    department?: string | null;
    recipient?: string | null;
    sent_at?: string | null;
    error?: string | null;
  } | null;
  department?: string | null;
  video_url?: string | null;
}

/** GET /api/analytics */
export interface Analytics {
  incident_counts: Record<string, number>;
  total_incidents: number;
}

/** GET /api/evidence */
export type EvidenceItem = {
  id: string;
  url?: string | null;
  path?: string | null;
  incident_id?: string | null;
  analysis_id?: string | null;
  video_filename?: string | null;
  classification: string;
  threat_level?: string | null;
  timestamp_seconds: number;
  window_index?: number | null;
  confidence?: number | null;
  kind: "frame" | "crop";
  department?: string | null;
  notification?: "sent" | "failed" | "not_required" | "unknown" | null;
  human_review?: {
    verdict: "correct" | "incorrect";
    corrected_label: string | null;
    notes: string | null;
    reviewed_at: string;
  } | null;
  created_at: string;
  source: "backend" | "client-capture";
  metadata?: Record<string, unknown>;
};
