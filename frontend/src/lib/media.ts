import { getApiBaseUrl } from "@/services/api";

/**
 * Turns a backend-provided media reference into a loadable URL.
 * Only values the backend actually returned are used — nothing is invented.
 * Returns null for absolute filesystem paths that cannot be served over HTTP.
 */
export function resolveMediaUrl(value: string | null | undefined): string | null {
  if (!value) return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (/^(https?:|blob:|data:)/i.test(trimmed)) return trimmed;
  if (trimmed.startsWith("/")) return `${getApiBaseUrl()}${trimmed}`;
  // Handle Windows-style backslashes returned by some Python path joins
  const normalized = trimmed.replace(/\\/g, "/");
  if (normalized.startsWith("api/")) return `${getApiBaseUrl()}/${normalized}`;
  // Windows / POSIX absolute filesystem paths are not HTTP-addressable.
  if (/^[a-zA-Z]:[\\/]/.test(trimmed) || trimmed.startsWith("\\\\") || trimmed.startsWith("/")) {
     // If it's an absolute path but contains 'dataset/preprocessed', try to convert to relative API URL
     const match = normalized.match(/dataset\/preprocessed\/(.+)$/);
     if (match) return `${getApiBaseUrl()}/api/evidence/static/${match[1]}`;
     return null;
  }
  return `${getApiBaseUrl()}/${normalized.replace(/^\.?\//, "")}`;
}

/** Formats a 0..1 or 0..100 confidence value. Returns null when absent. */
export function formatConfidence(value: number | null | undefined): string | null {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  const pct = value <= 1 ? value * 100 : value;
  return `${Math.round(pct)}%`;
}
