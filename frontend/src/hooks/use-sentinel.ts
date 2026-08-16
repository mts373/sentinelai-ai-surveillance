import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, api, validateVideoFile } from "@/services/api";
import type { AnalysisJob } from "@/types/sentinel";

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "Unexpected error.";
}

export function useSystemStatus(pollMs = 5000) {
  return useQuery({
    queryKey: ["system-status"],
    queryFn: () => api.systemStatus(),
    refetchInterval: pollMs,
    retry: false,
  });
}

export function useIncidents() {
  return useQuery({
    queryKey: ["incidents"],
    queryFn: () => api.incidents(),
    retry: false,
  });
}

export function useIncident(incidentId: string) {
  return useQuery({
    queryKey: ["incident", incidentId],
    queryFn: () => api.incident(incidentId),
    retry: false,
  });
}

export function useEvidence() {
  return useQuery({
    queryKey: ["evidence"],
    queryFn: () => api.evidence(),
    retry: false,
  });
}

export function useAnalytics() {
  return useQuery({
    queryKey: ["analytics"],
    queryFn: () => api.analytics(),
    retry: false,
  });
}

export function useIncidentActions(incidentId: string) {
  const queryClient = useQueryClient();
  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: ["incident", incidentId] });
    void queryClient.invalidateQueries({ queryKey: ["incidents"] });
    void queryClient.invalidateQueries({ queryKey: ["analytics"] });
  };

  const acknowledge = useMutation({
    mutationFn: () => api.acknowledgeIncident(incidentId),
    onSuccess: invalidate,
  });
  const resolve = useMutation({
    mutationFn: () => api.resolveIncident(incidentId),
    onSuccess: invalidate,
  });

  return { acknowledge, resolve };
}

const ANALYSIS_STORAGE_KEY = "sentinelai.lastAnalysisId";
const POLL_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 45 * 60 * 1000;

export function storeAnalysisId(id: string) {
  if (typeof window !== "undefined") window.localStorage.setItem(ANALYSIS_STORAGE_KEY, id);
}

export function loadStoredAnalysisId(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(ANALYSIS_STORAGE_KEY);
}

export interface UseVideoAnalysis {
  analysisId: string | null;
  job: AnalysisJob | null;
  uploading: boolean;
  uploadError: string | null;
  pollError: string | null;
  timedOut: boolean;
  start: (file: File) => Promise<void>;
  loadExisting: (id: string) => void;
  reset: () => void;
}

/**
 * Uploads a real file to POST /api/analyze-video, stores the returned
 * analysis_id and polls GET /api/analyze-video/{analysis_id}.
 * All state shown to the user comes from the backend response.
 */
export function useVideoAnalysis(): UseVideoAnalysis {
  const queryClient = useQueryClient();
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [timedOut, setTimedOut] = useState(false);
  const pollStartedAt = useRef<number | null>(null);

  const jobQuery = useQuery({
    queryKey: ["analysis", analysisId],
    queryFn: () => api.analysisStatus(analysisId as string),
    enabled: Boolean(analysisId) && !timedOut,
    retry: false,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (!status) return POLL_INTERVAL_MS;
      return status === "completed" || status === "failed" ? false : POLL_INTERVAL_MS;
    },
  });

  const job = jobQuery.data ?? null;
  const terminal = job?.status === "completed" || job?.status === "failed";

  useEffect(() => {
    if (!analysisId || terminal) {
      pollStartedAt.current = null;
      return;
    }
    if (pollStartedAt.current === null) pollStartedAt.current = Date.now();
    const timer = window.setInterval(() => {
      if (pollStartedAt.current && Date.now() - pollStartedAt.current > POLL_TIMEOUT_MS) {
        setTimedOut(true);
      }
    }, 5000);
    return () => window.clearInterval(timer);
  }, [analysisId, terminal]);

  useEffect(() => {
    if (job?.status === "completed") {
      void queryClient.invalidateQueries({ queryKey: ["incidents"] });
      void queryClient.invalidateQueries({ queryKey: ["analytics"] });
    }
  }, [job?.status, queryClient]);

  const start = useCallback(
    async (file: File) => {
      setUploadError(null);
      setTimedOut(false);
      const invalid = validateVideoFile(file);
      if (invalid) {
        setUploadError(invalid);
        return;
      }
      setUploading(true);
      try {
        const accepted = await api.analyzeVideo(file);
        storeAnalysisId(accepted.analysis_id);
        setAnalysisId(accepted.analysis_id);
      } catch (error) {
        setUploadError(errorMessage(error));
      } finally {
        setUploading(false);
      }
    },
    [],
  );

  const loadExisting = useCallback((id: string) => {
    setUploadError(null);
    setTimedOut(false);
    storeAnalysisId(id);
    setAnalysisId(id);
  }, []);

  const reset = useCallback(() => {
    setAnalysisId(null);
    setUploadError(null);
    setTimedOut(false);
  }, []);

  return {
    analysisId,
    job,
    uploading,
    uploadError,
    pollError: jobQuery.isError ? errorMessage(jobQuery.error) : null,
    timedOut,
    start,
    loadExisting,
    reset,
  };
}
