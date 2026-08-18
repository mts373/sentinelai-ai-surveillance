/**
 * Single API service layer for the real SentinelAI FastAPI backend.
 * No mock data, no fabricated fields.
 */
import type {
  Analytics,
  AnalysisJob,
  AnalyzeVideoAccepted,
  EvidenceItem,
  Incident,
  SystemStatus,
} from "@/types/sentinel";

const DEFAULT_BASE_URL =
  (import.meta.env["VITE_API_BASE_URL"] as string | undefined) ?? "http://127.0.0.1:8000";

const STORAGE_KEY = "sentinelai.apiBaseUrl";

export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    const override = window.localStorage.getItem(STORAGE_KEY);
    if (override && override.trim()) return override.trim().replace(/\/$/, "");
  }
  return DEFAULT_BASE_URL.replace(/\/$/, "");
}

export function setApiBaseUrl(url: string) {
  if (typeof window === "undefined") return;
  const trimmed = url.trim().replace(/\/$/, "");
  if (trimmed) window.localStorage.setItem(STORAGE_KEY, trimmed);
  else window.localStorage.removeItem(STORAGE_KEY);
}

export function getDefaultApiBaseUrl(): string {
  return DEFAULT_BASE_URL.replace(/\/$/, "");
}

export class ApiError extends Error {
  status: number | null;
  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const OFFLINE_MESSAGE =
  "Backend unavailable — make sure SentinelAI FastAPI is running on port 8000.";

async function readError(response: Response): Promise<string> {
  try {
    const text = await response.text();
    if (!text) return `Request failed with status ${response.status}.`;
    try {
      const data = JSON.parse(text) as { detail?: unknown };
      if (typeof data.detail === "string") return data.detail;
      return JSON.stringify(data.detail ?? data);
    } catch {
      return text.slice(0, 300);
    }
  } catch {
    return `Request failed with status ${response.status}.`;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}${path}`, init);
  } catch {
    throw new ApiError(OFFLINE_MESSAGE, null);
  }

  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }

  const text = await response.text();
  if (!text) throw new ApiError("Malformed response: empty body from backend.", response.status);
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError("Malformed response: backend did not return valid JSON.", response.status);
  }
}

export const api = {
  systemStatus: () => request<SystemStatus>("/api/system/status"),

  analyzeVideo: (file: File, signal?: AbortSignal) => {
    const body = new FormData();
    body.append("file", file);
    return request<AnalyzeVideoAccepted>("/api/analyze-video", {
      method: "POST",
      body,
      signal: signal ?? null,
    });
  },

  analysisStatus: (analysisId: string) =>
    request<AnalysisJob>(`/api/analyze-video/${encodeURIComponent(analysisId)}`),

  incidents: () => request<Incident[]>("/api/incidents"),

  incident: (incidentId: string) =>
    request<Incident>(`/api/incidents/${encodeURIComponent(incidentId)}`),

  acknowledgeIncident: (incidentId: string) =>
    request<Incident>(`/api/incidents/${encodeURIComponent(incidentId)}/acknowledge`, {
      method: "POST",
    }),

  resolveIncident: (incidentId: string) =>
    request<Incident>(`/api/incidents/${encodeURIComponent(incidentId)}/resolve`, {
      method: "POST",
    }),

  evidence: (filters?: { analysis_id?: string; incident_id?: string }) => {
    const params = new URLSearchParams();
    if (filters?.analysis_id) params.append("analysis_id", filters.analysis_id);
    if (filters?.incident_id) params.append("incident_id", filters.incident_id);
    const query = params.toString();
    return request<EvidenceItem[]>(`/api/evidence${query ? `?${query}` : ""}`);
  },

  saveEvidence: (evidence: Omit<EvidenceItem, "id" | "created_at">) =>
    request<EvidenceItem>("/api/evidence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(evidence),
    }),

  updateHumanReview: (
    analysisId: string,
    review: {
      verdict: "correct" | "incorrect";
      corrected_label: string | null;
      notes: string | null;
    },
  ) =>
    request<EvidenceItem>(`/api/analyze-video/${encodeURIComponent(analysisId)}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(review),
    }),


  analytics: () => request<Analytics>("/api/analytics"),
};

export const ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"];
export const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;

export function validateVideoFile(file: File): string | null {
  const lower = file.name.toLowerCase();
  const ok = ALLOWED_VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext));
  if (!ok) {
    return `Unsupported file type. Allowed: ${ALLOWED_VIDEO_EXTENSIONS.join(", ")}`;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return "File exceeds the backend limit of 500 MB.";
  }
  if (file.size === 0) {
    return "File is empty.";
  }
  return null;
}
