import { createFileRoute } from "@tanstack/react-router";
import { FilmIcon } from "lucide-react";

import { AppShell } from "@/components/app-shell";
import { EmptyState, ErrorNotice, LoadingRow, Panel } from "@/components/sentinel-ui";
import { errorMessage, useEvidence } from "@/hooks/use-sentinel";

export const Route = createFileRoute("/evidence")({
  head: () => ({
    meta: [
      { title: "Evidence — SentinelAI" },
      {
        name: "description",
        content:
          "Evidence clip library backed by the SentinelAI API, ready for generated incident clips once the pipeline exposes them.",
      },
      { property: "og:title", content: "Evidence — SentinelAI" },
      {
        property: "og:description",
        content: "Evidence clip library for SentinelAI incident investigations.",
      },
    ],
  }),
  component: EvidencePage,
});

function EvidencePage() {
  const evidence = useEvidence();
  const items = evidence.data ?? [];

  return (
    <AppShell title="Evidence" description="GET /api/evidence">
      <Panel title="Evidence Clips">
        {evidence.isLoading ? (
          <LoadingRow />
        ) : evidence.isError ? (
          <ErrorNotice message={errorMessage(evidence.error)} />
        ) : items.length === 0 ? (
          <EmptyState
            icon={FilmIcon}
            title="No evidence clips available."
            description="The inference pipeline does not expose generated evidence clips through the API yet. This view will list them as soon as the backend returns them."
          />
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {items.map((item, index) => {
              const url = typeof item.url === "string" ? item.url : null;
              const id = typeof item.id === "string" ? item.id : `clip-${index + 1}`;
              const incidentId = typeof item.incident_id === "string" ? item.incident_id : null;
              return (
                <div key={id} className="soc-panel overflow-hidden">
                  {url ? (
                    <video controls preload="metadata" className="aspect-video w-full bg-background">
                      <source src={url} />
                    </video>
                  ) : (
                    <div className="flex aspect-video items-center justify-center bg-background text-xs text-muted-foreground">
                      Clip URL not provided by backend
                    </div>
                  )}
                  <div className="border-t border-border px-3 py-2">
                    <div className="font-mono text-xs text-foreground">{id}</div>
                    {incidentId ? (
                      <div className="soc-label mt-0.5">Incident {incidentId}</div>
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Panel>
    </AppShell>
  );
}
