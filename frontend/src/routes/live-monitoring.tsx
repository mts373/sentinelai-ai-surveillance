import { createFileRoute } from "@tanstack/react-router";
import { MonitorOff, VideoOff } from "lucide-react";
import { useState } from "react";

import { AppShell } from "@/components/app-shell";
import { EmptyState, ErrorNotice, Panel } from "@/components/sentinel-ui";
import { errorMessage, useSystemStatus } from "@/hooks/use-sentinel";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/live-monitoring")({
  head: () => ({
    meta: [
      { title: "Live Monitoring — SentinelAI" },
      {
        name: "description",
        content:
          "Multi-camera monitoring wall for SentinelAI, ready for live stream integration once camera infrastructure is connected.",
      },
      { property: "og:title", content: "Live Monitoring — SentinelAI" },
      {
        property: "og:description",
        content: "Camera wall layouts prepared for future SentinelAI live stream integration.",
      },
    ],
  }),
  component: LiveMonitoringPage,
});

const LAYOUTS = [1, 4, 9] as const;

/** Placeholder tile. Renders no fake feed — awaits a real stream source. */
function CameraTile({ index }: { index: number }) {
  return (
    <div className="soc-grid-bg relative flex aspect-video items-center justify-center rounded-md border border-border bg-background">
      <div className="flex flex-col items-center gap-1.5 text-muted-foreground">
        <VideoOff className="h-5 w-5" />
        <span className="soc-label">Slot {index + 1}</span>
        <span className="text-xs">No stream source</span>
      </div>
    </div>
  );
}

function LiveMonitoringPage() {
  const status = useSystemStatus();
  const [layout, setLayout] = useState<(typeof LAYOUTS)[number]>(4);

  const cameras = status.data?.live_cameras ?? 0;

  return (
    <AppShell
      title="Live Monitoring"
      description="Camera wall — live stream integration pending"
      actions={
        <div className="flex items-center gap-1 rounded-md border border-border p-0.5">
          {LAYOUTS.map((option) => (
            <button
              key={option}
              onClick={() => setLayout(option)}
              className={cn(
                "rounded px-2.5 py-1 text-xs",
                layout === option
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {option === 1 ? "1×1" : option === 4 ? "2×2" : "3×3"}
            </button>
          ))}
        </div>
      }
    >
      <div className="space-y-4">
        {status.isError ? <ErrorNotice message={errorMessage(status.error)} /> : null}

        {cameras === 0 ? (
          <Panel title="Camera Status" subtitle="live_cameras from GET /api/system/status">
            <EmptyState
              icon={MonitorOff}
              title="No live cameras connected"
              description="The backend reports 0 live cameras and exposes no live stream endpoint. Camera slots below are layout placeholders and will render real feeds once a WebSocket or RTSP gateway is connected."
            />
          </Panel>
        ) : null}

        <Panel title={`Monitor Wall · ${layout === 1 ? "1" : layout} view`}>
          <div
            className={cn(
              "grid gap-3",
              layout === 1 ? "grid-cols-1" : layout === 4 ? "sm:grid-cols-2" : "sm:grid-cols-3",
            )}
          >
            {Array.from({ length: layout }).map((_, index) => (
              <CameraTile key={index} index={index} />
            ))}
          </div>
        </Panel>
      </div>
    </AppShell>
  );
}
