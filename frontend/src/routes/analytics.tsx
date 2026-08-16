import { createFileRoute } from "@tanstack/react-router";
import { BarChart3 } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { AppShell } from "@/components/app-shell";
import { EmptyState, ErrorNotice, LoadingRow, Metric, Panel } from "@/components/sentinel-ui";
import { errorMessage, useAnalytics } from "@/hooks/use-sentinel";

export const Route = createFileRoute("/analytics")({
  head: () => ({
    meta: [
      { title: "Analytics — SentinelAI" },
      {
        name: "description",
        content:
          "Incident analytics computed by the SentinelAI backend: totals and per-class distribution across Normal, Fire, Fight and Road Accident.",
      },
      { property: "og:title", content: "Analytics — SentinelAI" },
      {
        property: "og:description",
        content: "Incident distribution analytics from the SentinelAI backend.",
      },
    ],
  }),
  component: AnalyticsPage,
});

const CLASS_COLOR: Record<string, string> = {
  Normal: "var(--color-normal)",
  Fire: "var(--color-critical)",
  Fight: "var(--color-high)",
  "Road Accident": "var(--color-high)",
};

function AnalyticsPage() {
  const analytics = useAnalytics();
  const counts = analytics.data?.incident_counts ?? null;
  const data = counts
    ? Object.entries(counts).map(([name, value]) => ({ name, value }))
    : [];
  const hasData = data.some((entry) => entry.value > 0);

  return (
    <AppShell title="Analytics" description="GET /api/analytics">
      <div className="space-y-4">
        {analytics.isError ? <ErrorNotice message={errorMessage(analytics.error)} /> : null}

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Metric
            label="Total Incidents"
            value={analytics.data ? analytics.data.total_incidents : "—"}
          />
          {data.map((entry) => (
            <Metric key={entry.name} label={entry.name} value={entry.value} />
          ))}
        </div>

        <Panel title="Incident Distribution" subtitle="incident_counts">
          {analytics.isLoading ? (
            <LoadingRow />
          ) : !hasData ? (
            <EmptyState
              icon={BarChart3}
              title="No analytics data yet"
              description="The backend has recorded no incidents, so there is nothing to chart. Charts appear once real analyses produce incidents."
            />
          ) : (
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                  <CartesianGrid stroke="var(--color-border)" vertical={false} />
                  <XAxis
                    dataKey="name"
                    stroke="var(--color-muted-foreground)"
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis
                    allowDecimals={false}
                    stroke="var(--color-muted-foreground)"
                    tick={{ fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "var(--color-card)",
                      border: "1px solid var(--color-border)",
                      borderRadius: 6,
                      fontSize: 12,
                    }}
                  />
                  <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                    {data.map((entry) => (
                      <Cell
                        key={entry.name}
                        fill={CLASS_COLOR[entry.name] ?? "var(--color-chart-5)"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Panel>
      </div>
    </AppShell>
  );
}
