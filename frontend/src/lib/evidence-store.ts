/**
 * Local persistence for artefacts the FastAPI backend does not expose yet:
 * saved evidence frames/crops and human review decisions.
 *
 * Only metadata + small still images are kept — never video data.
 * AI predictions are never overwritten: a human correction is stored alongside.
 */
import { useCallback, useEffect, useState } from "react";
import { api } from "@/services/api";
import type { EvidenceItem, HumanReview as HumanReviewType, ReviewVerdict } from "@/types/sentinel";

const LOCAL_EVIDENCE_KEY = "sentinelai.localEvidence.v1";
const LOCAL_REVIEW_KEY = "sentinelai.localReviews.v1";
const EVENT = "sentinelai:store-change";

// Re-export for compatibility while we transition
export type EvidenceRecord = EvidenceItem;
export type HumanReview = HumanReviewType;

function readLocal<T>(key: string): T[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as T[]) : [];
  } catch {
    return [];
  }
}

function writeLocal<T>(key: string, value: T[]) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage full */
  }
  window.dispatchEvent(new Event(EVENT));
}

export function useEvidenceRecords(filter?: { analysisId?: string | null; incidentId?: string | null }) {
  const [backendEvidence, setBackendEvidence] = useState<EvidenceItem[]>([]);
  const [localEvidence, setLocalEvidence] = useState<EvidenceItem[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const remote = await api.evidence({
        analysis_id: filter?.analysisId || "",
        incident_id: filter?.incidentId || "",

      });
      setBackendEvidence(remote);
    } catch (e) {
      console.error("Failed to fetch backend evidence:", e);
    }

    const local = readLocal<EvidenceItem>(LOCAL_EVIDENCE_KEY).filter((item) => {
      if (filter?.analysisId && item.analysis_id !== filter.analysisId) return false;
      if (filter?.incidentId && item.incident_id !== filter.incidentId) return false;
      return true;
    });
    setLocalEvidence(local);
    setLoading(false);
  }, [filter?.analysisId, filter?.incidentId]);

  useEffect(() => {
    refresh();
    window.addEventListener(EVENT, refresh);
    return () => window.removeEventListener(EVENT, refresh);
  }, [refresh]);

  return { records: [...backendEvidence, ...localEvidence], loading, refresh };
}

export function useHumanReview(analysisId: string | null | undefined) {
  const [review, setReview] = useState<HumanReviewType | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!analysisId) return;
    setLoading(true);
    // Check backend first (via job status which includes review if implemented)
    try {
      const job = await api.analysisStatus(analysisId);
      if (job.result?.human_review) {
        setReview(job.result.human_review as HumanReviewType);
        setLoading(false);
        return;
      }
    } catch {}

    // Fallback to local
    const local = readLocal<HumanReviewType>(LOCAL_REVIEW_KEY).find(
      (r) => r.analysis_id === analysisId
    );
    setReview(local || null);
    setLoading(false);
  }, [analysisId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const submit = useCallback(
    async (input: Omit<HumanReviewType, "reviewed_at">) => {
      const fullReview = { ...input, reviewed_at: new Date().toISOString() };
      try {
        await api.updateHumanReview(input.analysis_id, {
          verdict: input.verdict,
          corrected_label: input.corrected_label,
          notes: input.notes,
        });
      } catch (e) {
        console.warn("Backend review update failed, saving locally:", e);
        const others = readLocal<HumanReviewType>(LOCAL_REVIEW_KEY).filter(
          (r) => r.analysis_id !== input.analysis_id
        );
        writeLocal(LOCAL_REVIEW_KEY, [fullReview, ...others]);
      }
      refresh();
    },
    [refresh]
  );

  return { review, submit, loading, refresh };
}

export async function saveEvidence(record: Omit<EvidenceItem, "id" | "created_at">) {
  try {
    return await api.saveEvidence(record);
  } catch (e) {
    console.warn("Backend evidence save failed, saving locally:", e);
    const localRecord: EvidenceItem = {
      ...record,
      id: `LOCAL-${Date.now()}`,
      created_at: new Date().toISOString(),
      source: "client-capture",
    };
    const current = readLocal<EvidenceItem>(LOCAL_EVIDENCE_KEY);
    writeLocal(LOCAL_EVIDENCE_KEY, [localRecord, ...current]);
    return localRecord;
  }
}


export function useAllReviews() {
  const [reviews, setReviews] = useState<HumanReviewType[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setLoading(true);
    setReviews(readLocal<HumanReviewType>(LOCAL_REVIEW_KEY));
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
    window.addEventListener(EVENT, refresh);
    return () => window.removeEventListener(EVENT, refresh);
  }, [refresh]);

  return { reviews, loading, refresh };
}

