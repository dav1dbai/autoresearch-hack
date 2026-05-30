import { readFileSync, writeFileSync } from "fs";

type DataPoint = {
  kernel: string;
  kernel_id: number;
  collection: string;
  category: string;
  submitted_at: string;
  sol_pct: number;
  username: string;
  best_sol_pct: number;
  is_new_best: boolean;
};

const data: DataPoint[] = JSON.parse(
  readFileSync(import.meta.dir + "/data/best_sol_over_time.json", "utf-8")
);

const COLLECTIONS = [
  { key: "L1", title: "L1 — Single Operations (94 kernels)" },
  { key: "L2", title: "L2 — Fused Operations (82 kernels)" },
  { key: "Quant", title: "Quantization (33 kernels)" },
  { key: "FlashInfer-Bench", title: "FlashInfer-Bench (26 kernels)" },
];

function makeSpec(collection: string, title: string, index: number) {
  const collData = data.filter((d) => d.collection === collection);
  const kernelNames = [...new Set(collData.map((d) => d.kernel))].sort();
  const paramName = `sel${index}`;

  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    title: { text: title, fontSize: 16, anchor: "start" as const },
    width: 900,
    height: Math.max(450, kernelNames.length * 7),
    data: { values: collData },
    params: [
      {
        name: paramName,
        select: { type: "point" as const, fields: ["kernel"] },
        bind: "legend",
      },
    ],
    mark: {
      type: "line" as const,
      interpolate: "step-after" as const,
      strokeWidth: 1.5,
      point: { filled: true, size: 50 },
    },
    encoding: {
      x: {
        field: "submitted_at",
        type: "temporal" as const,
        title: "Date",
        axis: { format: "%b %d", labelAngle: -45 },
      },
      y: {
        field: "sol_pct",
        type: "quantitative" as const,
        title: "% Speed of Light",
        scale: { domain: [0, 100] },
      },
      color: {
        field: "kernel",
        type: "nominal" as const,
        title: "Kernel",
        sort: kernelNames,
        legend: {
          orient: "right" as const,
          columns: 1,
          symbolSize: 80,
          labelLimit: 400,
          titleFontSize: 12,
          labelFontSize: 10,
        },
      },
      opacity: {
        condition: { param: paramName, value: 1 },
        value: 0.2,
      },
      strokeWidth: {
        condition: { param: paramName, value: 3 },
        value: 1.5,
      },
      tooltip: [
        { field: "kernel", type: "nominal" as const, title: "Kernel" },
        { field: "category", type: "nominal" as const, title: "Category" },
        {
          field: "submitted_at",
          type: "temporal" as const,
          title: "Date",
          format: "%Y-%m-%d %H:%M",
        },
        {
          field: "sol_pct",
          type: "quantitative" as const,
          title: "% SOL",
          format: ".1f",
        },
        { field: "username", type: "nominal" as const, title: "User" },
      ],
    },
    config: {
      view: { stroke: null },
      axis: { grid: true, gridColor: "#eee" },
    },
  };
}

const specs = COLLECTIONS.map((c, i) => makeSpec(c.key, c.title, i));

const totalKernels = new Set(data.map((d) => d.kernel)).size;
const totalPoints = data.filter((d) => d.is_new_best).length;

const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>SOL-ExecBench — % Speed of Light Over Time</title>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      margin: 0; padding: 24px;
      background: #f8f9fa;
      color: #1a1a1a;
    }
    h1 { margin: 0 0 4px; font-size: 28px; }
    .subtitle { color: #666; font-size: 14px; margin-bottom: 24px; }
    .subtitle a { color: #5a7; }
    .chart-section {
      background: #fff;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 24px;
      overflow-x: auto;
    }
  </style>
</head>
<body>
  <h1>SOL-ExecBench Leaderboard</h1>
  <p class="subtitle">
    Best % of theoretical speed-of-light over time per kernel, grouped by collection.
    Click a kernel in the legend to highlight it; hover points for details.
    <br>
    <strong>${totalKernels}</strong> kernels &bull;
    <strong>${totalPoints}</strong> improvement submissions &bull;
    Data from <a href="https://research.nvidia.com/benchmarks/sol-execbench/leaderboard">NVIDIA SOL-ExecBench</a>
  </p>

  ${COLLECTIONS.map(
    (_, i) => `<div class="chart-section"><div id="vis${i}"></div></div>`
  ).join("\n  ")}

  <script>
    const specs = ${JSON.stringify(specs)};
    const opts = {
      actions: { export: true, source: false, compiled: false, editor: false },
      renderer: 'canvas',
    };
    specs.forEach((spec, i) => {
      vegaEmbed('#vis' + i, spec, opts).catch(e => console.error('Chart ' + i + ':', e));
    });
  </script>
</body>
</html>`;

const outPath = import.meta.dir + "/sol_leaderboard.html";
writeFileSync(outPath, html);
console.log(`Wrote visualization to ${outPath}`);
