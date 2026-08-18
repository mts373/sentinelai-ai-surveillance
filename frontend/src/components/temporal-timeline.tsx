import { classificationClasses, formatSeconds } from "@/lib/sentinel";
import { formatConfidence } from "@/lib/media";
import { cn } from "@/lib/utils";
import type { AnalysisWindow } from "@/types/sentinel";

/**
 * Proportional timeline of the real backend temporal windows.
 * Clicking a segment seeks the player and selects the window.
 */
export function TemporalTimeline({
  windows,
  selected,
  currentTime,
  onSelect,
}: {
  windows: AnalysisWindow[];
  selected?: AnalysisWindow | null | undefined;
  currentTime?: number | undefined;
  onSelect: (win: AnalysisWindow) => void;
}) {
  if (windows.length === 0) return null;

  const start = Math.min(...windows.map((w) => w.start));
  const end = Math.max(...windows.map((w) => w.end));
  const span = Math.max(1, end - start);

  return (
    <div className="space-y-2">
      <div className="flex h-12 w-full overflow-hidden rounded-md border border-border">
        {windows.map((win) => {
          const c = classificationClasses(win.classification);
          const isActive = selected?.window === win.window;
          const isPlaying =
            currentTime !== undefined && currentTime >= win.start && currentTime < win.end;
          const width = ((win.end - win.start) / span) * 100;
          return (
            <button
              key={`seg-${win.window}-${win.start}`}
              type="button"
              onClick={() => onSelect(win)}
              style={{ width: `${width}%` }}
              title={`${formatSeconds(win.start)}–${formatSeconds(win.end)} · ${win.classification}`}
              className={cn(
                "group relative flex min-w-[2.5rem] flex-col items-center justify-center border-r border-border/60 px-1 transition-all last:border-r-0",
                c.bg,
                isActive && "ring-2 ring-inset ring-primary",
                isPlaying && "brightness-125",
              )}
            >
              <span className={cn("truncate text-[10px] font-semibold tracking-wide", c.text)}>
                {(win.classification ?? "?").toUpperCase()}
              </span>
              <span className="font-mono text-[9px] text-muted-foreground">
                {formatSeconds(win.start)}
              </span>
              {isPlaying ? (
                <span className="absolute inset-x-0 bottom-0 h-0.5 bg-primary" />
              ) : null}
            </button>
          );
        })}
      </div>
      <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
        <span>{formatSeconds(start)}</span>
        <span>{formatSeconds(end)}</span>
      </div>
      {selected ? (
        <div className="text-xs text-muted-foreground">
          Selected window #{selected.window} · {formatSeconds(selected.start)}–
          {formatSeconds(selected.end)}
          {formatConfidence(selected.confidence)
            ? ` · confidence ${formatConfidence(selected.confidence)}`
            : ""}
        </div>
      ) : null}
    </div>
  );
}
