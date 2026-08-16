import type { AnalysisJobStatus, ThreatLevel } from "@/types/sentinel";

export function threatClasses(level: ThreatLevel | null | undefined) {
  switch ((level ?? "").toUpperCase()) {
    case "CRITICAL":
      return {
        text: "text-critical",
        bg: "bg-critical/15",
        border: "border-critical/40",
        dot: "bg-critical",
      };
    case "HIGH":
      return {
        text: "text-high",
        bg: "bg-high/15",
        border: "border-high/40",
        dot: "bg-high",
      };
    case "LOW":
      return {
        text: "text-normal",
        bg: "bg-normal/15",
        border: "border-normal/40",
        dot: "bg-normal",
      };
    default:
      return {
        text: "text-muted-foreground",
        bg: "bg-muted",
        border: "border-border",
        dot: "bg-muted-foreground",
      };
  }
}

export function classificationClasses(classification: string | null | undefined) {
  switch (classification) {
    case "Fire":
      return threatClasses("CRITICAL");
    case "Fight":
    case "Road Accident":
      return threatClasses("HIGH");
    case "Normal":
      return threatClasses("LOW");
    default:
      return threatClasses(null);
  }
}

export function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const total = Math.max(0, Math.floor(value));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

export function formatDuration(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value < 60) return `${value.toFixed(1)}s`;
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes && bytes !== 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function formatTimestamp(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

/** Pipeline stages mapped from the real backend job status values. */
export const PIPELINE_STAGES = [
  { key: "upload", label: "Upload" },
  { key: "preprocessing", label: "Preprocessing" },
  { key: "temporal", label: "Temporal Analysis" },
  { key: "inference", label: "Qwen2.5-VL Inference" },
  { key: "aggregation", label: "Aggregation" },
  { key: "complete", label: "Complete" },
] as const;

export type PipelineStageState = "pending" | "active" | "done" | "failed";

/**
 * Derives stage states strictly from the backend job status.
 * No timers, no simulated progression.
 */
export function pipelineStageStates(
  status: AnalysisJobStatus | "uploading" | null,
): Record<string, PipelineStageState> {
  const order = ["upload", "preprocessing", "temporal", "inference", "aggregation", "complete"];
  const activeIndexByStatus: Record<string, number> = {
    uploading: 0,
    queued: 0,
    preprocessing: 1,
    inference: 3,
    aggregation: 4,
    completed: 5,
  };

  const states: Record<string, PipelineStageState> = {};
  for (const key of order) states[key] = "pending";
  if (!status) return states;

  if (status === "failed") {
    for (const key of order) states[key] = "pending";
    states["upload"] = "done";
    states["complete"] = "failed";
    return states;
  }

  const activeIndex = activeIndexByStatus[status];
  if (activeIndex === undefined) return states;

  order.forEach((key, index) => {
    if (index < activeIndex) states[key] = "done";
    else if (index === activeIndex) states[key] = status === "completed" ? "done" : "active";
  });

  // Temporal analysis is performed by the preprocessor/inference bridge; it is
  // complete once the backend reports the inference stage.
  if (status === "inference" || status === "aggregation" || status === "completed") {
    states["temporal"] = "done";
  }
  if (status === "completed") {
    for (const key of order) states[key] = "done";
  }
  return states;
}
