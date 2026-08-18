import { createFileRoute, Link } from "@tanstack/react-router";
import {
  CheckCircle2,
  Circle,
  Loader2,
  RotateCcw,
  Upload,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { VideoPlayer, type VideoPlayerHandle } from "@/components/video-player";
import { TemporalTimeline } from "@/components/temporal-timeline";
import { resolveMediaUrl } from "@/lib/media";


import { AppShell } from "@/components/app-shell";
import {
  ClassBadge,
  ErrorNotice,
  Panel,
  ThreatBadge,
  EmptyState,
} from "@/components/sentinel-ui";
import { loadStoredAnalysisId, useVideoAnalysis } from "@/hooks/use-sentinel";
import {
  PIPELINE_STAGES,
  formatBytes,
  formatDuration,
  formatSeconds,
  formatTimestamp,
  pipelineStageStates,
} from "@/lib/sentinel";
import { ALLOWED_VIDEO_EXTENSIONS } from "@/services/api";
import { cn } from "@/lib/utils";
import type { AnalysisWindow } from "@/types/sentinel";

export const Route = createFileRoute("/video-analysis")({
  head: () => ({
    meta: [
      { title: "Video Analysis — SentinelAI" },
      {
        name: "description",
        content:
          "Upload CCTV footage to the SentinelAI backend and follow the real Qwen2.5-VL temporal analysis pipeline from preprocessing to aggregation.",
      },
      { property: "og:title", content: "Video Analysis — SentinelAI" },
      {
        property: "og:description",
        content:
          "Run real Qwen2.5-VL incident analysis on CCTV footage and review temporal detection windows.",
      },
    ],
  }),
  component: VideoAnalysisPage,
});

function StageRow({
  label,
  state,
}: {
  label: string;
  state: "pending" | "active" | "done" | "failed";
}) {
  const Icon: LucideIcon =
    state === "done" ? CheckCircle2 : state === "failed" ? XCircle : state === "active" ? Loader2 : Circle;
  const color =
    state === "done"
      ? "text-normal"
      : state === "failed"
        ? "text-critical"
        : state === "active"
          ? "text-primary"
          : "text-muted-foreground/50";
  return (
    <div className="flex items-center gap-2.5 py-1.5">
      <Icon className={cn("h-4 w-4 shrink-0", color, state === "active" && "animate-spin")} />
      <span
        className={cn(
          "text-xs tracking-wide",
          state === "pending" ? "text-muted-foreground" : "text-foreground",
        )}
      >
        {label.toUpperCase()}
      </span>
    </div>
  );
}

function VideoAnalysisPage() {
  const analysis = useVideoAnalysis();
  const inputRef = useRef<HTMLInputElement>(null);
  const [selected, setSelected] = useState<File | null>(null);
  const [selectedWindow, setSelectedWindow] = useState<AnalysisWindow | null>(null);
  const [restorable, setRestorable] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const videoPlayerRef = useRef<VideoPlayerHandle>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);


  useEffect(() => {
    setRestorable(loadStoredAnalysisId());
  }, []);

  const job = analysis.job;
  const result = job?.result ?? null;
  const stageState = pipelineStageStates(
    analysis.uploading ? "uploading" : (job?.status ?? null),
  );

  useEffect(() => {
    setSelectedWindow(null);
    if (result?.video?.url) {
      setVideoUrl(resolveMediaUrl(result.video.url));
    } else if (result?.video?.path) {
      setVideoUrl(resolveMediaUrl(result.video.path));
    }
  }, [job?.analysis_id, job?.status, result]);

  useEffect(() => {
    if (selected) {
      const url = URL.createObjectURL(selected);
      setVideoUrl(url);
      return () => URL.revokeObjectURL(url);
    }
    return undefined;
  }, [selected]);

  const handleWindowSelect = useCallback((win: AnalysisWindow) => {
    setSelectedWindow(win);
    videoPlayerRef.current?.seek(win.start, true);
  }, []);



  const onSubmit = async () => {
    if (!selected) return;
    await analysis.start(selected);
  };

  return (
    <AppShell
      title="Video Analysis"
      description="Upload → POST /api/analyze-video → poll GET /api/analyze-video/{analysis_id}"
      actions={
        analysis.analysisId ? (
          <button
            onClick={() => {
              analysis.reset();
              setSelected(null);
              if (inputRef.current) inputRef.current.value = "";
            }}
            className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-xs text-foreground hover:bg-accent"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            New analysis
          </button>
        ) : null
      }
    >
      <div className="grid gap-4 xl:grid-cols-[minmax(0,20rem)_minmax(0,1fr)]">
        <div className="space-y-4">
          <Panel title="Source Video">
            <label
              htmlFor="video-file"
              className="flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-border px-4 py-8 text-center hover:border-primary/60"
            >
              <Upload className="h-6 w-6 text-muted-foreground" />
              <span className="mt-2 text-sm text-foreground">
                {selected ? selected.name : "Select a CCTV video"}
              </span>
              <span className="mt-1 text-xs text-muted-foreground">
                {selected
                  ? formatBytes(selected.size)
                  : `${ALLOWED_VIDEO_EXTENSIONS.join(" · ")} · max 500 MB`}
              </span>
            </label>
            <input
              id="video-file"
              ref={inputRef}
              type="file"
              accept={ALLOWED_VIDEO_EXTENSIONS.join(",")}
              className="sr-only"
              onChange={(event) => setSelected(event.target.files?.[0] ?? null)}
            />

            <button
              onClick={onSubmit}
              disabled={!selected || analysis.uploading}
              className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {analysis.uploading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Uploading to backend…
                </>
              ) : (
                "Start analysis"
              )}
            </button>

            {analysis.uploadError ? (
              <ErrorNotice className="mt-3" message={analysis.uploadError} />
            ) : null}

            {!analysis.analysisId && restorable ? (
              <button
                onClick={() => analysis.loadExisting(restorable)}
                className="mt-3 w-full rounded-md border border-border px-3 py-1.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
              >
                Resume last analysis · <span className="font-mono">{restorable}</span>
              </button>
            ) : null}
          </Panel>

          <Panel title="Processing Pipeline" subtitle="Derived from backend job status">
            <div className="divide-y divide-border/60">
              {PIPELINE_STAGES.map((stage) => (
                <StageRow
                  key={stage.key}
                  label={stage.label}
                  state={stageState[stage.key] ?? "pending"}
                />
              ))}
            </div>
            {job ? (
              <div className="mt-3 space-y-2 border-t border-border pt-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="soc-label">Backend progress</span>
                  <span className="font-mono text-foreground">{job.progress}%</span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded bg-muted">
                  <div
                    className={cn(
                      "h-full rounded",
                      job.status === "failed" ? "bg-critical" : "bg-primary",
                    )}
                    style={{ width: `${Math.max(0, Math.min(100, job.progress))}%` }}
                  />
                </div>
                {job.stage ? (
                  <div className="text-xs text-foreground">{job.stage}</div>
                ) : null}
                {job.message ? (
                  <p className="text-xs text-muted-foreground">{job.message}</p>
                ) : null}
                {job.last_log ? (
                  <pre className="max-h-28 overflow-auto rounded border border-border bg-background p-2 font-mono text-[11px] text-muted-foreground">
                    {job.last_log}
                  </pre>
                ) : null}
              </div>
            ) : null}
          </Panel>
        </div>

        <div className="space-y-4">
          {videoUrl ? (
            <Panel title="Analysis Player" subtitle={selected?.name || job?.filename || ""}>
              <VideoPlayer
                ref={videoPlayerRef}
                src={videoUrl}
                filename={selected?.name || job?.filename}
                onTime={setCurrentTime}
              />
              {result?.windows && (
                <div className="mt-4 border-t border-border pt-4">
                  <div className="soc-label mb-2">Temporal localization</div>
                  <TemporalTimeline
                    windows={result.windows}
                    selected={selectedWindow}
                    currentTime={currentTime}
                    onSelect={handleWindowSelect}
                  />
                </div>
              )}
            </Panel>
          ) : null}

          {analysis.pollError ? <ErrorNotice message={analysis.pollError} /> : null}


          {!analysis.analysisId ? (
            <Panel title="Result">
              <EmptyState
                title="No analysis running"
                description="Select a video and start an analysis. Results are produced entirely by the local SentinelAI backend."
              />
            </Panel>
          ) : null}

          {analysis.analysisId && !result ? (
            <Panel title="Analysis" subtitle={analysis.analysisId}>
              <dl className="grid gap-3 sm:grid-cols-2">
                <div>
                  <dt className="soc-label">Status</dt>
                  <dd className="text-sm text-foreground">
                    {(job?.status ?? "pending").toUpperCase()}
                  </dd>
                </div>
                <div>
                  <dt className="soc-label">Filename</dt>
                  <dd className="truncate text-sm text-foreground">{job?.filename ?? "—"}</dd>
                </div>
                <div>
                  <dt className="soc-label">Uploaded size</dt>
                  <dd className="text-sm text-foreground">{formatBytes(job?.size_bytes)}</dd>
                </div>
                <div>
                  <dt className="soc-label">Created</dt>
                  <dd className="text-sm text-foreground">{formatTimestamp(job?.created_at)}</dd>
                </div>
              </dl>
            </Panel>
          ) : null}

          {result ? (
            <>
              <Panel title="Final Detection" subtitle={result.analysis_id ?? analysis.analysisId ?? ""}>
                <div className="flex flex-wrap items-center gap-3">
                  <div className="text-3xl font-semibold text-foreground">
                    {result.final_classification}
                  </div>
                  <ThreatBadge level={result.threat_level} />
                </div>
                <dl className="mt-4 grid gap-3 sm:grid-cols-2">
                  <div>
                    <dt className="soc-label">Video</dt>
                    <dd className="break-all text-sm text-foreground">{result.video.filename}</dd>
                  </div>
                  <div>
                    <dt className="soc-label">Model</dt>
                    <dd className="text-sm text-foreground">{result.model}</dd>
                  </div>
                  <div>
                    <dt className="soc-label">Analysis ID</dt>
                    <dd className="font-mono text-xs text-foreground">
                      {result.analysis_id ?? analysis.analysisId}
                    </dd>
                  </div>
                  {result.processing_time !== null && result.processing_time !== undefined ? (
                    <div>
                      <dt className="soc-label">Processing time</dt>
                      <dd className="text-sm text-foreground">
                        {formatDuration(result.processing_time)}
                      </dd>
                    </div>
                  ) : null}
                </dl>
                <div className="mt-4 space-y-3 border-t border-border pt-3">
                  <div>
                    <div className="soc-label">AI summary</div>
                    <p className="mt-1 text-sm text-foreground">{result.summary}</p>
                  </div>
                  <div>
                    <div className="soc-label">Recommended action</div>
                    <p className="mt-1 text-sm text-foreground">{result.recommended_action}</p>
                  </div>
                  {result.incident_id ? (
                    <Link
                      to="/incidents/$incidentId"
                      params={{ incidentId: result.incident_id }}
                      className="inline-flex items-center gap-1.5 text-xs font-medium text-primary hover:underline"
                    >
                      Open full incident report {result.incident_id}
                    </Link>
                  ) : (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <CheckCircle2 className="h-3.5 w-3.5 text-normal" />
                      Normal activity confirmed · No incident record created
                    </div>
                  )}

                </div>
              </Panel>

              <Panel
                title="Temporal Analysis"
                subtitle={`${result.windows.length} inference windows returned by the backend`}
              >
                {result.windows.length === 0 ? (
                  <EmptyState title="No temporal windows returned" />
                ) : (
                  <div className="space-y-1.5">
                    {result.windows.map((win) => {
                      const active = selectedWindow?.window === win.window;
                      const isCurrent = currentTime >= win.start && currentTime < win.end;
                      return (
                        <button
                          key={`${win.window}-${win.start}`}
                          onClick={() => handleWindowSelect(win)}
                          className={cn(
                            "flex w-full flex-wrap items-center gap-3 rounded-md border px-3 py-2 text-left transition-colors",
                            active
                              ? "border-primary/60 bg-accent/50"
                              : "border-border hover:bg-accent/30",
                            isCurrent && !active && "border-primary/30",
                          )}
                        >
                          <span className="font-mono text-xs text-muted-foreground">
                            {formatSeconds(win.start)} ───── {formatSeconds(win.end)}
                          </span>
                          <ClassBadge value={win.classification} />
                          {win.error ? (
                            <span className="text-xs text-critical">window error</span>
                          ) : null}
                          {isCurrent && (
                            <span className="ml-2 h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
                          )}
                          <span className="ml-auto font-mono text-[11px] text-muted-foreground">
                            #{win.window}
                          </span>
                        </button>
                      );
                    })}
                  </div>

                )}

                {selectedWindow ? (
                  <div className="mt-4 space-y-4">
                    <div className="space-y-2.5 rounded-md border border-border bg-background p-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-xs text-muted-foreground">
                          Window #{selectedWindow.window} · {formatSeconds(selectedWindow.start)}–
                          {formatSeconds(selectedWindow.end)}
                        </span>
                        <ClassBadge value={selectedWindow.classification} />
                      </div>
                      {selectedWindow.evidence ? (
                        <div>
                          <div className="soc-label">Evidence</div>
                          <p className="mt-0.5 text-sm text-foreground">{selectedWindow.evidence}</p>
                        </div>
                      ) : null}
                      {selectedWindow.incident_summary ? (
                        <div>
                          <div className="soc-label">Incident summary</div>
                          <p className="mt-0.5 text-sm text-foreground">
                            {selectedWindow.incident_summary}
                          </p>
                        </div>
                      ) : null}
                      {selectedWindow.processing_time !== null &&
                      selectedWindow.processing_time !== undefined ? (
                        <div className="text-xs text-muted-foreground">
                          Inference time: {formatDuration(selectedWindow.processing_time)}
                        </div>
                      ) : null}
                      {selectedWindow.error ? <ErrorNotice message={selectedWindow.error} /> : null}
                    </div>
                    {/* Evidence Gallery Placeholder - would be populated if backend returned frame URLs */}
                    {selectedWindow.frames && selectedWindow.frames.length > 0 && (
                      <div className="grid grid-cols-2 gap-2">
                        {selectedWindow.frames.map((f, i) => (
                          <div key={i} className="group relative aspect-video overflow-hidden rounded border border-border bg-muted">
                            <img 
                              src={resolveMediaUrl(f) || ""} 
                              alt={`Evidence ${i}`} 
                              className="h-full w-full object-cover transition-transform group-hover:scale-105"
                            />
                            <div className="absolute inset-x-0 bottom-0 bg-black/60 p-1 text-[10px] text-white">
                              {formatSeconds(selectedWindow.start)}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : null}

              </Panel>


              {result.window_counts ? (
                <Panel title="Window Classification Counts">
                  <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                    {Object.entries(result.window_counts).map(([label, count]) => (
                      <div
                        key={label}
                        className="rounded-md border border-border px-3 py-2"
                      >
                        <div className="soc-label">{label}</div>
                        <div className="text-lg font-semibold tabular-nums text-foreground">
                          {count}
                        </div>
                      </div>
                    ))}
                  </div>
                </Panel>
              ) : null}
            </>
          ) : null}
        </div>
      </div>
    </AppShell>
  );
}
