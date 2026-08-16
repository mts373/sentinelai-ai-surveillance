import { createFileRoute, Link } from "@tanstack/react-router";
import { AlertTriangle, Camera, Clock, Cpu, FileVideo, Shield } from "lucide-react";

import { AppShell } from "@/components/app-shell";
import {
  ClassBadge,
  EmptyState,
  ErrorNotice,
  GpuPanelBody,
  LoadingRow,
  Metric,
  Panel,
  ThreatBadge,
} from "@/components/sentinel-ui";
import { errorMessage, useAnalytics, useIncidents, useSystemStatus } from "@/hooks/use-sentinel";
import { formatTimestamp } from "@/lib/sentinel";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "SentinelAI — Incident Intelligence Overview" },
      {
        name: "description",
        content:
          "SentinelAI security operations overview: backend status, GPU telemetry, active incidents and Qwen2.5-VL analysis activity.",
      },
      { property: "og:title", content: "SentinelAI — Incident Intelligence Overview" },
      {
        property: "og:description",
        content:
          "Security operations overview for SentinelAI: live backend status, GPU telemetry and active incidents.",
      },
    ],
  }),
  component: OverviewPage,
});

function OverviewPage() {
  const status = useSystemStatus();
  const incidents = useIncidents();
  const analytics = useAnalytics();

  const backendDown = status.isError;
  const openIncidents = (incidents.data ?? []).filter((i) => i.status !== "RESOLVED");

  return (
    <AppShell
      title="Overview"
      description="Operational state reported by the SentinelAI backend"
      actions={
        <Link
          to="/video-analysis"
          className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
        >
          <FileVideo className="h-3.5 w-3.5" />
          Analyze video
        </Link>
      }
    >
      <div className="space-y-4">
        {backendDown ? <ErrorNotice message={errorMessage(status.error)} /> : null}

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Metric
            label="Active Cameras"
            value={status.data ? status.data.live_cameras : "—"}
            hint={
              status.data && status.data.live_cameras === 0
                ? "Live camera infrastructure not connected"
                : undefined
            }
          />
          <Metric
            label="Active Incidents"
            value={incidents.data ? openIncidents.length : "—"}
            tone={openIncidents.length > 0 ? "bad" : undefined}
            hint={
              incidents.data
                ? `${incidents.data.length} total in backend memory`
                : "Awaiting backend"
            }
          />
          <Metric
            label="Total Incidents"
            value={analytics.data ? analytics.data.total_incidents : "—"}
            hint="From /api/analytics"
          />
          <Metric
            label="System Status"
            value={status.data ? status.data.ai_engine.toUpperCase() : "NOT CONNECTED"}
            tone={status.data ? (status.data.ai_engine === "ready" ? "ok" : "warn") : "bad"}
            hint={status.data ? `Checked ${formatTimestamp(status.data.timestamp)}` : undefined}
          />
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          <Panel title="GPU" subtitle="GET /api/system/status" className="lg:col-span-1">
            {status.isLoading && !status.data ? (
              <LoadingRow />
            ) : status.data ? (
              <GpuPanelBody gpu={status.data.gpu} />
            ) : (
              <p className="text-sm text-muted-foreground">Backend not connected.</p>
            )}
          </Panel>

          <Panel title="AI Engine" className="lg:col-span-1">
            <dl className="space-y-2.5 text-sm">
              <div className="flex items-start justify-between gap-3">
                <dt className="soc-label">Model</dt>
                <dd className="text-right font-mono text-xs text-foreground">
                  {status.data?.model ?? "—"}
                </dd>
              </div>
              <div className="flex items-center justify-between gap-3">
                <dt className="soc-label">Engine</dt>
                <dd className="text-foreground">
                  {status.data ? status.data.ai_engine.toUpperCase() : "NOT CONNECTED"}
                </dd>
              </div>
              <div className="flex items-center justify-between gap-3">
                <dt className="soc-label">Analysis running</dt>
                <dd className="text-foreground">
                  {status.data ? (status.data.active_analysis ? "Yes" : "No") : "—"}
                </dd>
              </div>
            </dl>
            <div className="mt-4 space-y-1.5 border-t border-border pt-3">
              <div className="soc-label">Detection classes</div>
              <div className="flex flex-wrap gap-1.5">
                <ClassBadge value="Normal" />
                <ClassBadge value="Fire" />
                <ClassBadge value="Fight" />
                <ClassBadge value="Road Accident" />
              </div>
              <p className="pt-1 text-xs text-muted-foreground">
                Unauthorized Entry — planned, not implemented in the current engine.
              </p>
            </div>
          </Panel>

          <Panel title="Coverage" className="lg:col-span-1">
            <div className="space-y-3 text-sm">
              <div className="flex items-start gap-2.5">
                <Camera className="mt-0.5 h-4 w-4 text-muted-foreground" />
                <div>
                  <div className="text-foreground">Live monitoring</div>
                  <p className="text-xs text-muted-foreground">
                    No live camera endpoint is exposed by the backend yet.
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-2.5">
                <Cpu className="mt-0.5 h-4 w-4 text-muted-foreground" />
                <div>
                  <div className="text-foreground">Local inference</div>
                  <p className="text-xs text-muted-foreground">
                    Analyses execute on the local backend host, one job at a time.
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-2.5">
                <Shield className="mt-0.5 h-4 w-4 text-muted-foreground" />
                <div>
                  <div className="text-foreground">Uploaded video analysis</div>
                  <p className="text-xs text-muted-foreground">
                    Incidents are created only when a real analysis is not classified Normal.
                  </p>
                </div>
              </div>
            </div>
          </Panel>
        </div>

        <Panel
          title="Recent Incidents"
          subtitle="GET /api/incidents"
          actions={
            <Link to="/incidents" className="text-xs text-primary hover:underline">
              View all
            </Link>
          }
        >
          {incidents.isLoading ? (
            <LoadingRow />
          ) : incidents.isError ? (
            <ErrorNotice message={errorMessage(incidents.error)} />
          ) : (incidents.data ?? []).length === 0 ? (
            <EmptyState
              icon={AlertTriangle}
              title="No incidents recorded"
              description="The backend has not produced any non-Normal classification yet. Run a video analysis to generate real incidents."
            />
          ) : (
            <ul className="divide-y divide-border">
              {(incidents.data ?? []).slice(0, 6).map((incident) => (
                <li key={incident.id}>
                  <Link
                    to="/incidents/$incidentId"
                    params={{ incidentId: incident.id }}
                    className="flex flex-wrap items-center gap-3 px-1 py-2.5 hover:bg-accent/40"
                  >
                    <span className="font-mono text-xs text-muted-foreground">{incident.id}</span>
                    <ClassBadge value={incident.incident_type} />
                    <ThreatBadge level={incident.threat_level} />
                    <span className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {formatTimestamp(incident.date_time)}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </AppShell>
  );
}
