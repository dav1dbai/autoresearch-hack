const API = "https://research.nvidia.com/benchmarks/sol-execbench/api";
const CONCURRENCY = 10;
const OUT = import.meta.dir + "/data";

type Kernel = {
  id: number;
  name: string;
  submission_count: number;
  tags: string[];
};
type Ranking = {
  rank: number | null;
  username: string;
  sol_score: number;
  latency_ms: number;
  submitted_at?: string;
  is_reference?: boolean;
};
type LeaderboardResponse = {
  data: {
    kernel_id: number;
    kernel_title: string;
    baseline_latency_ms: number;
    rankings: Record<string, Ranking[]>;
  };
};

const COLLECTION_TAGS = ["L1", "L2", "Quant", "FlashInfer-Bench"] as const;
const CATEGORY_TAGS = [
  "attention",
  "normalization",
  "rope",
  "moe",
  "mlp",
  "vision",
  "decoder",
  "diffusion",
  "ssm",
  "audio",
  "video",
  "gemm",
  "mla_paged_attention",
  "quantization",
  "other",
] as const;

function classifyKernel(tags: string[]) {
  const collection =
    COLLECTION_TAGS.find((c) => tags.includes(c)) ?? "Unknown";
  const category =
    CATEGORY_TAGS.find((c) => tags.includes(c)) ?? "other";
  return { collection, category };
}

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.json();
}

async function main() {
  const { mkdirSync, writeFileSync } = await import("fs");
  mkdirSync(OUT, { recursive: true });

  console.log("Fetching kernel list...");
  const { data } = await fetchJSON<{ data: { kernels: Kernel[] } }>(
    `${API}/kernels`
  );
  const kernels = data.kernels.sort((a, b) => a.id - b.id);
  console.log(`Found ${kernels.length} kernels`);

  const kernelMeta = new Map<
    number,
    { collection: string; category: string; tags: string[] }
  >();
  for (const k of kernels) {
    const { collection, category } = classifyKernel(k.tags);
    kernelMeta.set(k.id, { collection, category, tags: k.tags });
  }

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

  const allPoints: DataPoint[] = [];
  let done = 0;
  const queue = [...kernels];

  async function worker() {
    while (queue.length > 0) {
      const kernel = queue.shift()!;
      const meta = kernelMeta.get(kernel.id)!;
      try {
        const lb = await fetchJSON<LeaderboardResponse>(
          `${API}/leaderboard/${kernel.id}?gpu_type=B200`
        );
        const rankings = lb.data.rankings["B200"] ?? [];
        const submissions = rankings
          .filter((r) => !r.is_reference && r.submitted_at)
          .sort(
            (a, b) =>
              new Date(a.submitted_at!).getTime() -
              new Date(b.submitted_at!).getTime()
          );

        let bestSol = 0;
        for (const sub of submissions) {
          const isNewBest = sub.sol_score > bestSol;
          if (isNewBest) bestSol = sub.sol_score;
          allPoints.push({
            kernel: lb.data.kernel_title,
            kernel_id: kernel.id,
            collection: meta.collection,
            category: meta.category,
            submitted_at: sub.submitted_at!,
            sol_pct: +(sub.sol_score * 100).toFixed(2),
            username: sub.username,
            best_sol_pct: +(bestSol * 100).toFixed(2),
            is_new_best: isNewBest,
          });
        }
      } catch (e) {
        console.error(`Failed kernel ${kernel.id}: ${e}`);
      }
      done++;
      if (done % 20 === 0) console.log(`  ${done}/${kernels.length}`);
    }
  }

  const workers = Array.from({ length: CONCURRENCY }, () => worker());
  await Promise.all(workers);
  console.log(`  ${done}/${kernels.length} done`);

  allPoints.sort(
    (a, b) =>
      new Date(a.submitted_at).getTime() - new Date(b.submitted_at).getTime()
  );

  writeFileSync(
    `${OUT}/all_submissions.json`,
    JSON.stringify(allPoints, null, 2)
  );
  console.log(
    `Wrote ${allPoints.length} data points to ${OUT}/all_submissions.json`
  );

  // Build best-over-time per kernel
  const bestTimeSeries: DataPoint[] = [];
  const currentBest = new Map<string, number>();

  for (const p of allPoints) {
    const cur = currentBest.get(p.kernel) ?? 0;
    if (p.sol_pct > cur) {
      currentBest.set(p.kernel, p.sol_pct);
      bestTimeSeries.push({ ...p, is_new_best: true });
    }
  }

  // Extend each kernel's last best to "now"
  const now = new Date().toISOString();
  const latestByKernel = new Map<string, DataPoint>();
  for (const p of bestTimeSeries) {
    latestByKernel.set(p.kernel, p);
  }
  for (const [, last] of latestByKernel) {
    bestTimeSeries.push({
      ...last,
      submitted_at: now,
      is_new_best: false,
    });
  }

  bestTimeSeries.sort(
    (a, b) =>
      new Date(a.submitted_at).getTime() - new Date(b.submitted_at).getTime()
  );

  writeFileSync(
    `${OUT}/best_sol_over_time.json`,
    JSON.stringify(bestTimeSeries, null, 2)
  );
  console.log(
    `Wrote ${bestTimeSeries.length} best-over-time points to ${OUT}/best_sol_over_time.json`
  );

  // Summary
  const collectionCounts = new Map<string, number>();
  for (const [, meta] of kernelMeta) {
    collectionCounts.set(
      meta.collection,
      (collectionCounts.get(meta.collection) ?? 0) + 1
    );
  }
  console.log("\nKernels per collection:");
  for (const [c, n] of collectionCounts) {
    console.log(`  ${c}: ${n}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
