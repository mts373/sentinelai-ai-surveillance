import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowLeft, VideoOff } from "lucide-react";

import { AppShell } from "@/components/app-shell";
import {
  ClassBadge,
  ErrorNotice,
  LoadingRow,
  Panel,
  ThreatBadge,
} from "@/components/sentinel-ui";
import { errorMessage, useIncident, useIncidentActions } from "@/hooks/use-sentinel";
import { formatTimestamp } from "@/lib/sentinel";

export const Route = createFileRoute("/incidents/$incidentId")({
  head: () => ({
    meta: [
      { title: "Incident Detail — SentinelAI" },
      {
        name: "description",
        content:
          "Full incident record from the SentinelAI backend: classification, threat level, AI summary, recommended action and originating analysis.",
      },
      { property: "og:title", content: "Incident Detail — SentinelAI" },
      {
        property: "og:description",
        content: "Full SentinelAI incident record with AI summary and recommended action.",
      },
    ],
  }),
  component: IncidentDetailPage,
});

function IncidentDetailPage() {
  const { incidentId } = Route.useParams();
  const incident = useIncident(incidentId);
  const { acknowledge, resolve } = useIncidentActions(incidentId);

  return (
    <AppShell
      title={`Incident ${incidentId}`}
      description="GET /api/incidents/{incident_id}"
      actions={
        <Link
          to="/incidents"
          className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-xs text-foreground hover:bg-accent"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back
        </Link>
      }
    >
      {incident.isLoading ? (
        <LoadingRow />
      ) : incident.isError ? (
        <ErrorNotice message={errorMessage(incident.error)} />
      ) : incident.data ? (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,20rem)]">
          <div className="space-y-4">
            <Panel title="Classification">
              <div className="flex flex-wrap items-center gap-3">
                <div className="text-3xl font-semibold text-foreground">
                  {incident.data.incident_type}
                </div>
                <ThreatBadge level={incident.data.threat_level} />
                <span className="rounded border border-border bg-muted px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
                  {incident.data.status}
                </span>
              </div>
              <dl className="mt-4 grid gap-3 sm:grid-cols-2">
                <div>
                  <dt className="soc-label">Timestamp</dt>
                  <dd className="text-sm text-foreground">
                    {formatTimestamp(incident.data.date_time)}
                  </dd>
                </div>
                <div>
                  <dt className="soc-label">Location</dt>
                  <dd className="text-sm text-foreground">{incident.data.location}</dd>
                </div>
                <div>
                  <dt className="soc-label">Camera</dt>
                  <dd className="text-sm text-foreground">
                    {incident.data.camera_id ?? "Not attached to a camera"}
                  </dd>
                </div>
                <div>
                  <dt className="soc-label">Analysis ID</dt>
                  <dd className="break-all font-mono text-xs text-foreground">
                    {incident.data.analysis_id}
                  </dd>
                </div>
              </dl>
            </Panel>

            <Panel title="AI Summary">
              <p className="text-sm text-foreground">{incident.data.summary}</p>
              <div className="mt-4 border-t border-border pt-3">
                <div className="soc-label">Recommended action</div>
                <p className="mt-1 text-sm text-foreground">{incident.data.recommended_action}</p>
              </div>
            </Panel>

            <Panel title="Evidence">
              <div className="flex items-center gap-2.5 text-sm text-muted-foreground">
                <VideoOff className="h-4 w-4" />
                Evidence clip not available — the backend does not expose evidence clips yet.
              </div>
            </Panel>
          </div>

          <div className="space-y-4">
            <Panel title="Triage">
              <div className="space-y-2">
                <button
                  onClick={() => acknowledge.mutate()}
                  disabled={acknowledge.isPending || incident.data.status !== "NEW"}
                  className="w-full rounded-md border border-border px-3 py-2 text-sm text-foreground hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {acknowledge.isPending ? "Acknowledging…" : "Acknowledge"}
                </button>
                <button
                  onClick={() => resolve.mutate()}
                  disabled={resolve.isPending || incident.data.status === "RESOLVED"}
                  className="w-full rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {resolve.isPending ? "Resolving…" : "Resolve"}
                </button>
              </div>
              {acknowledge.isError ? (
                <ErrorNotice className="mt-3" message={errorMessage(acknowledge.error)} />
              ) : null}
              {resolve.isError ? (
                <ErrorNotice className="mt-3" message={errorMessage(resolve.error)} />
              ) : null}
            </Panel>

            <Panel title="Detection Class">
              <ClassBadge value={incident.data.incident_type} />
              <p className="mt-3 text-xs text-muted-foreground">
                Produced by Qwen2.5-VL-7B-Instruct with the SentinelAI LoRA adapter.
              </p>
            </Panel>
          </div>
        </div>
      ) : null}
    </AppShell>
  );
}
