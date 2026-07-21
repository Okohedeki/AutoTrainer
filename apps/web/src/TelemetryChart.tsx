import { useMemo, useState } from "react";

export type ChartPoint = { x: number; y: number };
export type ChartSeries = { id: string; label: string; color: string; points: ChartPoint[] };

function safeRange(min: number, max: number) {
  if (min !== max) return { min, max };
  const padding = Math.abs(min) > 0 ? Math.abs(min) * 0.1 : 1;
  return { min: min - padding, max: max + padding };
}

// This small SVG renderer plots only observed backend values. It deliberately
// avoids interpolation and prediction so a sparse local run still reads honestly.
export default function TelemetryChart({
  title,
  description,
  series,
  emptyMessage,
  fixedY,
}: {
  title: string;
  description: string;
  series: ChartSeries[];
  emptyMessage: string;
  fixedY?: { min: number; max: number };
}) {
  const [hidden, setHidden] = useState<Set<string>>(() => new Set());
  const visible = series.filter((item) => !hidden.has(item.id) && item.points.length > 0);
  const allPoints = visible.flatMap((item) => item.points);

  const bounds = useMemo(() => {
    if (allPoints.length === 0) return null;
    const x = safeRange(Math.min(...allPoints.map((point) => point.x)), Math.max(...allPoints.map((point) => point.x)));
    const y = fixedY ?? safeRange(Math.min(...allPoints.map((point) => point.y)), Math.max(...allPoints.map((point) => point.y)));
    return { x, y };
  }, [allPoints, fixedY]);

  const width = 720;
  const height = 260;
  const plot = { left: 52, right: 18, top: 16, bottom: 35 };
  const plotWidth = width - plot.left - plot.right;
  const plotHeight = height - plot.top - plot.bottom;
  const scaleX = (value: number) => bounds ? plot.left + ((value - bounds.x.min) / (bounds.x.max - bounds.x.min)) * plotWidth : plot.left;
  const scaleY = (value: number) => bounds ? plot.top + (1 - ((value - bounds.y.min) / (bounds.y.max - bounds.y.min))) * plotHeight : plot.top;

  const toggle = (id: string) => {
    setHidden((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <section className="telemetry-chart" aria-label={title}>
      <header>
        <div><h3>{title}</h3><p>{description}</p></div>
        <div className="chart-legend" aria-label={`${title} series`}>
          {series.map((item) => {
            const latest = item.points.at(-1);
            const isHidden = hidden.has(item.id);
            return (
              <button
                type="button"
                key={item.id}
                className={isHidden ? "muted" : ""}
                onClick={() => toggle(item.id)}
                aria-pressed={!isHidden}
                disabled={item.points.length === 0}
              >
                <i style={{ background: item.color }} aria-hidden="true" />
                <span>{item.label}</span>
                <code>{latest ? latest.y.toFixed(3) : "No data"}</code>
              </button>
            );
          })}
        </div>
      </header>

      {!bounds ? (
        <div className="chart-empty"><strong>Waiting for observed values</strong><p>{emptyMessage}</p></div>
      ) : (
        <svg className="chart-canvas" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title}. ${visible.length} visible series from ${allPoints.length} observed values.`}>
          {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
            const y = plot.top + fraction * plotHeight;
            const value = bounds.y.max - fraction * (bounds.y.max - bounds.y.min);
            return (
              <g key={fraction}>
                <line x1={plot.left} x2={width - plot.right} y1={y} y2={y} className="chart-grid-line" />
                <text x={plot.left - 9} y={y + 3} textAnchor="end" className="chart-axis-label">{value.toFixed(2)}</text>
              </g>
            );
          })}
          <line x1={plot.left} x2={width - plot.right} y1={height - plot.bottom} y2={height - plot.bottom} className="chart-axis-line" />
          <text x={plot.left} y={height - 10} className="chart-axis-label">{Math.round(bounds.x.min)}</text>
          <text x={width - plot.right} y={height - 10} textAnchor="end" className="chart-axis-label">{Math.round(bounds.x.max)}</text>
          {visible.map((item) => {
            const coordinates = item.points.map((point) => `${scaleX(point.x)},${scaleY(point.y)}`).join(" ");
            return (
              <g key={item.id}>
                {item.points.length > 1 && <polyline points={coordinates} fill="none" stroke={item.color} strokeWidth={item.id === "reward" ? 3 : 2} strokeLinejoin="round" strokeLinecap="round" />}
                {item.points.map((point, index) => (
                  <circle key={`${point.x}-${index}`} cx={scaleX(point.x)} cy={scaleY(point.y)} r={item.points.length === 1 ? 4 : 2.5} fill={item.color}>
                    <title>{`${item.label}: ${point.y.toFixed(4)} at ${point.x}`}</title>
                  </circle>
                ))}
              </g>
            );
          })}
        </svg>
      )}
    </section>
  );
}
