import { createFileRoute } from "@tanstack/react-router";
import { FilmIcon, Video } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { AppShell } from "@/components/app-shell";
import { EmptyState, ErrorNotice, LoadingRow, Panel } from "@/components/sentinel-ui";
import { errorMessage } from "@/hooks/use-sentinel";
import { useEvidenceRecords } from "@/lib/evidence-store";
import { resolveMediaUrl } from "@/lib/media";
import { ClassBadge } from "@/components/sentinel-ui";
import { formatTimestamp } from "@/lib/sentinel";


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
  const { records: items, loading: isLoading } = useEvidenceRecords();


  return (
    <AppShell title="Evidence" description="GET /api/evidence">
      <Panel title="Evidence Clips">
        {isLoading ? (
          <LoadingRow />
        ) : items.length === 0 ? (
          <EmptyState
            icon={FilmIcon}
            title="No evidence available"
            description="Analyzed video evidence frames and manual captures will appear here once available from the backend or captured during analysis."
          />
        ) : (

          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {items.map((item) => {
              const url = resolveMediaUrl(item.url || item.path);
              return (
                <div key={item.id} className="soc-panel overflow-hidden flex flex-col">
                  <div className="relative aspect-video w-full bg-background group">
                    {url ? (
                      <img 
                        src={url} 
                        alt="Evidence" 
                        className="h-full w-full object-cover transition-transform group-hover:scale-105" 
                      />
                    ) : (
                      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
                        No image source
                      </div>
                    )}
                    <div className="absolute inset-x-0 bottom-0 bg-black/60 p-2 text-[10px] text-white backdrop-blur-sm flex justify-between">
                      <span>{item.classification.toUpperCase()}</span>
                      <span>{item.timestamp_seconds}s</span>
                    </div>
                  </div>
                  <div className="p-3 space-y-2 flex-1">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[10px] text-muted-foreground">{item.id}</span>
                      <ClassBadge value={item.classification} />
                    </div>
                    <div className="text-[11px] text-muted-foreground">
                      Captured: {formatTimestamp(item.created_at)}
                    </div>
                    {item.incident_id && (
                      <Link
                        to="/incidents/$incidentId"
                        params={{ incidentId: item.incident_id }}
                        className="flex items-center gap-1.5 text-xs text-primary hover:underline pt-1"
                      >
                        <Video className="h-3 w-3" />
                        View Incident {item.incident_id}
                      </Link>
                    )}
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
