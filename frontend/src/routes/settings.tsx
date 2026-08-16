import { createFileRoute } from "@tanstack/react-router";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { useEffect, useState } from "react";

import { AppShell } from "@/components/app-shell";
import { ErrorNotice, GpuPanelBody, Panel } from "@/components/sentinel-ui";
import { errorMessage, useSystemStatus } from "@/hooks/use-sentinel";
import { formatTimestamp } from "@/lib/sentinel";
import { api, getApiBaseUrl, getDefaultApiBaseUrl, setApiBaseUrl } from "@/services/api";
import type { SystemStatus } from "@/types/sentinel";

export const Route = createFileRoute("/settings")({
  head: () => ({
    meta: [
      { title: "Settings — SentinelAI" },
      {
        name: "description",
        content:
          "Configure the SentinelAI backend URL and verify backend, AI engine, model and GPU status with a live connection test.",
      },
      { property: "og:title", content: "Settings — SentinelAI" },
      {
        property: "og:description",
        content: "Configure the SentinelAI backend URL and test the live connection.",
      },
    ],
  }),
  component: SettingsPage,
});

type TestState =
  | { kind: "idle" }
  | { kind: "testing" }
  | { kind: "ok"; status: SystemStatus }
  | { kind: "error"; message: string };

function SettingsPage() {
  const status = useSystemStatus();
  const [baseUrl, setBaseUrl] = useState("");
  const [test, setTest] = useState<TestState>({ kind: "idle" });

  useEffect(() => {
    setBaseUrl(getApiBaseUrl());
  }, []);

  const runTest = async () => {
    setTest({ kind: "testing" });
    setApiBaseUrl(baseUrl);
    try {
      const result = await api.systemStatus();
      setTest({ kind: "ok", status: result });
      void status.refetch();
    } catch (error) {
      setTest({ kind: "error", message: errorMessage(error) });
    }
  };

  const live = test.kind === "ok" ? test.status : (status.data ?? null);

  return (
    <AppShell title="Settings" description="Backend connection and engine information">
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Backend">
          <label className="block">
            <span className="soc-label">Backend URL</span>
            <input
              value={baseUrl}
              onChange={(event) => setBaseUrl(event.target.value)}
              spellCheck={false}
              className="mt-1 h-9 w-full rounded-md border border-input bg-background px-2.5 font-mono text-sm text-foreground outline-none focus:border-ring"
              placeholder={getDefaultApiBaseUrl()}
            />
          </label>
          <p className="mt-1.5 text-xs text-muted-foreground">
            Default from VITE_API_BASE_URL: <span className="font-mono">{getDefaultApiBaseUrl()}</span>
          </p>

          <div className="mt-3 flex flex-wrap gap-2">
            <button
              onClick={runTest}
              disabled={test.kind === "testing"}
              className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            >
              {test.kind === "testing" ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              Test Connection
            </button>
            <button
              onClick={() => {
                setApiBaseUrl("");
                setBaseUrl(getDefaultApiBaseUrl());
                setTest({ kind: "idle" });
              }}
              className="rounded-md border border-border px-3 py-2 text-sm text-foreground hover:bg-accent"
            >
              Reset
            </button>
          </div>

          {test.kind === "ok" ? (
            <div className="mt-3 flex items-center gap-2 rounded-md border border-normal/40 bg-normal/10 px-3 py-2 text-sm text-foreground">
              <CheckCircle2 className="h-4 w-4 text-normal" />
              GET /api/system/status responded at {formatTimestamp(test.status.timestamp)}
            </div>
          ) : null}
          {test.kind === "error" ? <ErrorNotice className="mt-3" message={test.message} /> : null}

          <dl className="mt-4 space-y-2.5 border-t border-border pt-3 text-sm">
            <div className="flex items-center justify-between gap-3">
              <dt className="soc-label">Connection status</dt>
              <dd className="flex items-center gap-1.5 text-foreground">
                {live ? (
                  <>
                    <CheckCircle2 className="h-3.5 w-3.5 text-normal" /> CONNECTED
                  </>
                ) : (
                  <>
                    <XCircle className="h-3.5 w-3.5 text-critical" /> DISCONNECTED
                  </>
                )}
              </dd>
            </div>
            <div className="flex items-center justify-between gap-3">
              <dt className="soc-label">AI engine</dt>
              <dd className="text-foreground">
                {live ? live.ai_engine.toUpperCase() : "NOT CONNECTED"}
              </dd>
            </div>
            <div className="flex items-start justify-between gap-3">
              <dt className="soc-label">Model</dt>
              <dd className="text-right font-mono text-xs text-foreground">{live?.model ?? "—"}</dd>
            </div>
            <div className="flex items-center justify-between gap-3">
              <dt className="soc-label">Live cameras</dt>
              <dd className="text-foreground">{live ? live.live_cameras : "—"}</dd>
            </div>
          </dl>
        </Panel>

        <div className="space-y-4">
          <Panel title="GPU">
            <GpuPanelBody gpu={live?.gpu} />
          </Panel>

          <Panel title="Detection Capability">
            <ul className="space-y-1.5 text-sm text-foreground">
              <li>Normal</li>
              <li>Fire</li>
              <li>Fight</li>
              <li>Road Accident</li>
              <li className="text-muted-foreground">Unauthorized Entry — planned</li>
            </ul>
          </Panel>
        </div>
      </div>
    </AppShell>
  );
}
