import { createFileRoute, Link } from "@tanstack/react-router";
import { AlertTriangle } from "lucide-react";
import { useMemo, useState } from "react";

import { AppShell } from "@/components/app-shell";
import {
  ClassBadge,
  EmptyState,
  ErrorNotice,
  LoadingRow,
  Panel,
  ThreatBadge,
} from "@/components/sentinel-ui";
import { errorMessage, useIncidents } from "@/hooks/use-sentinel";
import { formatTimestamp } from "@/lib/sentinel";

export const Route = createFileRoute("/incidents/")({
  head: () => ({
    meta: [
      { title: "Incidents — SentinelAI" },
      {
        name: "description",
        content:
          "Review and triage incidents created by real SentinelAI analyses, filtered by type, threat level, status and location.",
      },
      { property: "og:title", content: "Incidents — SentinelAI" },
      {
        property: "og:description",
        content: "Triage real SentinelAI incidents by type, threat level, status and location.",
      },
    ],
  }),
  component: IncidentsPage,
});

const ANY = "__any__";

function Filter({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="soc-label">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="h-8 rounded-md border border-input bg-background px-2 text-xs text-foreground"
      >
        <option value={ANY}>All</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function IncidentsPage() {
  const incidents = useIncidents();
  const [type, setType] = useState(ANY);
  const [threat, setThreat] = useState(ANY);
  const [status, setStatus] = useState(ANY);
  const [location, setLocation] = useState(ANY);

  const rows = incidents.data ?? [];

  const options = useMemo(
    () => ({
      type: [...new Set(rows.map((r) => r.incident_type))],
      threat: [...new Set(rows.map((r) => r.threat_level))],
      status: [...new Set(rows.map((r) => r.status))],
      location: [...new Set(rows.map((r) => r.location))],
    }),
    [rows],
  );

  const filtered = rows.filter(
    (row) =>
      (type === ANY || row.incident_type === type) &&
      (threat === ANY || row.threat_level === threat) &&
      (status === ANY || row.status === status) &&
      (location === ANY || row.location === location),
  );

  return (
    <AppShell
      title="Incidents"
      description={`GET /api/incidents · ${rows.length} record${rows.length === 1 ? "" : "s"}`}
    >
      <Panel
        title="Incident Queue"
        actions={
          rows.length > 0 ? (
            <div className="flex flex-wrap items-end gap-2">
              <Filter label="Type" value={type} options={options.type} onChange={setType} />
              <Filter label="Threat" value={threat} options={options.threat} onChange={setThreat} />
              <Filter label="Status" value={status} options={options.status} onChange={setStatus} />
              <Filter
                label="Location"
                value={location}
                options={options.location}
                onChange={setLocation}
              />
            </div>
          ) : null
        }
      >
        {incidents.isLoading ? (
          <LoadingRow />
        ) : incidents.isError ? (
          <ErrorNotice message={errorMessage(incidents.error)} />
        ) : rows.length === 0 ? (
          <EmptyState
            icon={AlertTriangle}
            title="No incidents"
            description="The backend has no incidents. An incident is created only when an analysis returns a non-Normal classification."
          />
        ) : filtered.length === 0 ? (
          <EmptyState title="No incidents match the selected filters" />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left">
                  {["Incident", "Threat", "Time", "Status", "Location", "Analysis ID"].map((h) => (
                    <th key={h} className="soc-label px-2 pb-2 font-normal">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((incident) => (
                  <tr key={incident.id} className="border-b border-border/60 hover:bg-accent/30">
                    <td className="px-2 py-2.5">
                      <Link
                        to="/incidents/$incidentId"
                        params={{ incidentId: incident.id }}
                        className="flex flex-wrap items-center gap-2"
                      >
                        <span className="font-mono text-xs text-muted-foreground">
                          {incident.id}
                        </span>
                        <ClassBadge value={incident.incident_type} />
                      </Link>
                    </td>
                    <td className="px-2 py-2.5">
                      <ThreatBadge level={incident.threat_level} />
                    </td>
                    <td className="whitespace-nowrap px-2 py-2.5 text-xs text-muted-foreground">
                      {formatTimestamp(incident.date_time)}
                    </td>
                    <td className="px-2 py-2.5 font-mono text-xs text-foreground">
                      {incident.status}
                    </td>
                    <td className="px-2 py-2.5 text-xs text-foreground">{incident.location}</td>
                    <td className="px-2 py-2.5 font-mono text-[11px] text-muted-foreground">
                      {incident.analysis_id}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </AppShell>
  );
}
