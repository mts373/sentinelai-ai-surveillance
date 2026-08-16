import { AlertCircle, Inbox, Loader2 } from "lucide-react";
import type { ReactNode } from "react";

import { classificationClasses, threatClasses } from "@/lib/sentinel";
import { cn } from "@/lib/utils";
import type { GpuStatus, ThreatLevel } from "@/types/sentinel";

export function Panel({
  title,
  subtitle,
  actions,
  className,
  children,
}: {
  title?: string | undefined;
  subtitle?: string | undefined;
  actions?: ReactNode;
  className?: string | undefined;
  children: ReactNode;
}) {
  return (
    <section className={cn("soc-panel", className)}>
      {title ? (
        <header className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2.5">
          <div className="min-w-0">
            <h2 className="soc-label">{title}</h2>
            {subtitle ? (
              <p className="mt-0.5 truncate text-xs text-muted-foreground">{subtitle}</p>
            ) : null}
          </div>
          {actions ? <div className="ml-auto flex items-center gap-2">{actions}</div> : null}
        </header>
      ) : null}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function ThreatBadge({ level }: { level: ThreatLevel | null | undefined }) {
  const c = threatClasses(level);
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded border px-2 py-0.5 font-mono text-[11px] tracking-wider",
        c.text,
        c.bg,
        c.border,
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", c.dot)} />
      {(level ?? "UNKNOWN").toString().toUpperCase()}
    </span>
  );
}

export function ClassBadge({ value }: { value: string | null | undefined }) {
  const c = classificationClasses(value);
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-xs font-medium",
        c.text,
        c.bg,
        c.border,
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", c.dot)} />
      {value ?? "Unknown"}
    </span>
  );
}

export function ErrorNotice({
  message,
  className,
}: {
  message: string;
  className?: string | undefined;
}) {
  return (
    <div
      className={cn(
        "flex items-start gap-2 rounded-md border border-critical/40 bg-critical/10 px-3 py-2.5 text-sm text-foreground",
        className,
      )}
      role="alert"
    >
      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-critical" />
      <span className="min-w-0 break-words">{message}</span>
    </div>
  );
}

export function EmptyState({
  title,
  description,
  icon: Icon = Inbox,
  action,
}: {
  title: string;
  description?: string | undefined;
  icon?: React.ComponentType<{ className?: string }> | undefined;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-md border border-dashed border-border px-6 py-12 text-center">
      <Icon className="h-7 w-7 text-muted-foreground" />
      <p className="mt-3 text-sm font-medium text-foreground">{title}</p>
      {description ? (
        <p className="mt-1 max-w-md text-xs text-muted-foreground">{description}</p>
      ) : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  );
}

export function LoadingRow({ label = "Loading from backend…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 px-1 py-6 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" />
      {label}
    </div>
  );
}

export function Metric({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: ReactNode;
  hint?: string | undefined;
  tone?: "ok" | "warn" | "bad" | undefined;
}) {
  const toneClass =
    tone === "ok"
      ? "text-normal"
      : tone === "warn"
        ? "text-high"
        : tone === "bad"
          ? "text-critical"
          : "text-foreground";
  return (
    <div className="soc-panel p-4">
      <div className="soc-label">{label}</div>
      <div className={cn("mt-1.5 text-2xl font-semibold tabular-nums", toneClass)}>{value}</div>
      {hint ? <div className="mt-1 text-xs text-muted-foreground">{hint}</div> : null}
    </div>
  );
}

/** Renders the GPU object field-by-field. Never renders the object as a child. */
export function GpuPanelBody({ gpu }: { gpu: GpuStatus | null | undefined }) {
  if (!gpu) {
    return <p className="text-sm text-muted-foreground">GPU information unavailable.</p>;
  }
  if (!gpu.available) {
    return <p className="text-sm text-high">CUDA GPU not available on the backend host.</p>;
  }

  const used = gpu.memory_used_gb;
  const total = gpu.memory_total_gb;
  const pct = used !== null && total ? Math.min(100, (used / total) * 100) : null;

  return (
    <div className="space-y-3">
      <div className="text-sm font-medium text-foreground">{gpu.name ?? "GPU detected"}</div>
      <div>
        <div className="flex items-baseline justify-between text-xs">
          <span className="soc-label">Memory</span>
          <span className="font-mono text-foreground">
            {used !== null && total !== null ? `${used} / ${total} GB` : "Not reported"}
          </span>
        </div>
        {pct !== null ? (
          <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded bg-muted">
            <div className="h-full rounded bg-primary" style={{ width: `${pct}%` }} />
          </div>
        ) : null}
      </div>
      <div className="flex items-baseline justify-between text-xs">
        <span className="soc-label">Utilization</span>
        <span className="font-mono text-foreground">
          {gpu.utilization_percent !== null ? `${gpu.utilization_percent}%` : "Not reported"}
        </span>
      </div>
    </div>
  );
}
