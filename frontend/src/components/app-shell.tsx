import { Link } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Cpu,
  FileVideo,
  LayoutDashboard,
  Menu,
  Radio,
  Settings,
  Shield,
  Video,
  X,
} from "lucide-react";
import { useState, type ReactNode } from "react";

import { useSystemStatus } from "@/hooks/use-sentinel";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/live-monitoring", label: "Live Monitoring", icon: Radio },
  { to: "/video-analysis", label: "Video Analysis", icon: FileVideo },
  { to: "/incidents", label: "Incidents", icon: AlertTriangle },
  { to: "/evidence", label: "Evidence", icon: Video },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/settings", label: "Settings", icon: Settings },
] as const;

function StatusChip({
  label,
  value,
  tone,
  icon: Icon,
}: {
  label: string;
  value: string;
  tone: "ok" | "warn" | "bad" | "idle";
  icon: React.ComponentType<{ className?: string }>;
}) {
  const dot =
    tone === "ok"
      ? "bg-normal"
      : tone === "warn"
        ? "bg-high"
        : tone === "bad"
          ? "bg-critical"
          : "bg-muted-foreground";

  return (
    <div className="flex items-center gap-2 border-l border-border px-3 py-1 first:border-l-0 first:pl-0">
      <Icon className="h-3.5 w-3.5 text-muted-foreground" />
      <div className="leading-tight">
        <div className="soc-label">{label}</div>
        <div className="flex items-center gap-1.5 text-xs font-medium text-foreground">
          <span className={cn("h-1.5 w-1.5 rounded-full", dot)} />
          <span className="max-w-[13rem] truncate">{value}</span>
        </div>
      </div>
    </div>
  );
}

function TopStatusBar() {
  const { data, isError, isLoading } = useSystemStatus();

  const backendValue = isLoading && !data ? "CHECKING" : data ? "CONNECTED" : "DISCONNECTED";
  const backendTone = data ? "ok" : isError ? "bad" : "idle";

  const engine = data?.ai_engine?.toUpperCase();
  const engineValue = data ? (engine ?? "UNKNOWN") : "NOT CONNECTED";
  const engineTone = !data ? "bad" : engine === "READY" ? "ok" : engine === "BUSY" ? "warn" : "idle";

  const gpu = data?.gpu;
  const gpuValue = !data
    ? "UNAVAILABLE"
    : gpu?.available
      ? (gpu.name ?? "GPU detected")
      : "NOT AVAILABLE";
  const gpuTone = !data ? "bad" : gpu?.available ? "ok" : "warn";

  return (
    <div className="flex flex-wrap items-center gap-x-1 gap-y-2 border-b border-border bg-card px-4 py-2">
      <StatusChip label="Backend" value={backendValue} tone={backendTone} icon={Activity} />
      <StatusChip label="AI Engine" value={engineValue} tone={engineTone} icon={Shield} />
      <StatusChip label="GPU" value={gpuValue} tone={gpuTone} icon={Cpu} />
      {data ? (
        <div className="ml-auto hidden items-center gap-2 md:flex">
          <span className="soc-label">Model</span>
          <span className="font-mono text-xs text-foreground">{data.model}</span>
        </div>
      ) : null}
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <>
      <div className="flex h-14 items-center gap-2 border-b border-sidebar-border px-4">
        <Shield className="h-5 w-5 text-primary" />
        <div className="leading-tight">
          <div className="text-sm font-semibold tracking-wide text-sidebar-foreground">
            SentinelAI
          </div>
          <div className="soc-label">Incident Intelligence</div>
        </div>
      </div>
      <nav className="flex-1 space-y-0.5 overflow-y-auto p-2">
        {NAV.map((item) => (
          <Link
            key={item.to}
            to={item.to}
            onClick={onNavigate}
            activeOptions={{ exact: item.to === "/" }}
            className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
            activeProps={{
              className:
                "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm bg-sidebar-accent text-sidebar-accent-foreground border-l-2 border-primary",
            }}
          >
            <item.icon className="h-4 w-4" />
            {item.label}
          </Link>
        ))}
      </nav>
      <div className="border-t border-sidebar-border p-3">
        <div className="soc-label">AI Classes</div>
        <div className="mt-1.5 flex flex-wrap gap-1 text-[11px]">
          <span className="rounded border border-normal/40 bg-normal/15 px-1.5 py-0.5 text-normal">
            Normal
          </span>
          <span className="rounded border border-critical/40 bg-critical/15 px-1.5 py-0.5 text-critical">
            Fire
          </span>
          <span className="rounded border border-high/40 bg-high/15 px-1.5 py-0.5 text-high">
            Fight
          </span>
          <span className="rounded border border-high/40 bg-high/15 px-1.5 py-0.5 text-high">
            Road Accident
          </span>
          <span className="rounded border border-border bg-muted px-1.5 py-0.5 text-muted-foreground">
            Unauthorized Entry · Planned
          </span>
        </div>
      </div>
    </>
  );
}

export function AppShell({
  title,
  description,
  actions,
  children,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-background">
      <aside className="hidden w-60 shrink-0 flex-col border-r border-sidebar-border bg-sidebar lg:flex">
        <SidebarContent />
      </aside>

      {open ? (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button
            aria-label="Close navigation"
            className="absolute inset-0 bg-background/80"
            onClick={() => setOpen(false)}
          />
          <aside className="relative flex h-full w-60 flex-col border-r border-sidebar-border bg-sidebar">
            <button
              aria-label="Close navigation"
              className="absolute right-2 top-3 rounded p-1 text-muted-foreground hover:text-foreground"
              onClick={() => setOpen(false)}
            >
              <X className="h-4 w-4" />
            </button>
            <SidebarContent onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      ) : null}

      <div className="flex min-w-0 flex-1 flex-col">
        <TopStatusBar />
        <header className="flex flex-wrap items-center gap-3 border-b border-border bg-background px-4 py-3">
          <button
            aria-label="Open navigation"
            className="rounded border border-border p-1.5 text-muted-foreground hover:text-foreground lg:hidden"
            onClick={() => setOpen(true)}
          >
            <Menu className="h-4 w-4" />
          </button>
          <div className="min-w-0">
            <h1 className="truncate text-lg font-semibold text-foreground">{title}</h1>
            {description ? (
              <p className="truncate text-xs text-muted-foreground">{description}</p>
            ) : null}
          </div>
          {actions ? <div className="ml-auto flex items-center gap-2">{actions}</div> : null}
        </header>
        <main className="min-w-0 flex-1 p-4">{children}</main>
      </div>
    </div>
  );
}
