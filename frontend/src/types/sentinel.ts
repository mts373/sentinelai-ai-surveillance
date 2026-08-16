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
export interface AnalysisWindow {
  window: number;
  start: number;
  end: number;
  classification: SentinelClass | string;
  evidence: string;
  incident_summary: string;
  processing_time: number | null;
  error: string | null;
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
  video: { filename: string; path: string };
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
}

/** GET /api/analytics */
export interface Analytics {
  incident_counts: Record<string, number>;
  total_incidents: number;
}

/** GET /api/evidence — backend currently returns [] */
export type EvidenceItem = {
  id?: string;
  url?: string;
  incident_id?: string;
  [key: string]: unknown;
};
