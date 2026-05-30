"""cuda-oxide kernel-optimization envs — the inner agent hill-climbs a Rust->PTX
kernel toward throughput, scored on the deployed `ar2-cudaoxide` Modal app (H100).

Multi-kernel by design: each target (gemm, reduction, ...) is a `KernelSpec` holding
ONLY the client-side editable surface (starter/naive kernel, prompt, problem size,
target). The fixed host harness (timing + correctness oracle) lives server-side in
app.py, keyed by `kernel_name` — the agent never sees or ships it. The contract
between env and grader is the `kernel_name` string, so `CudaOxideEnv.__init__` takes
only JSON-primitive args and survives the Modal env_to_spec round-trip.

Conforms to `harness.contracts.Env`, so it drops into the outer loop like any env.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from harness.contracts import Split, StepResult, Submission, TaskSpec

_HERE = Path(__file__).parent
_KERNELS_DIR = _HERE / "kernels"
_APP = "ar2-cudaoxide"


@dataclass(frozen=True)
class KernelSpec:
    """Client-side description of one optimizable kernel. The matching fixed-host
    grader lives in app.py under the same `name`."""

    name: str
    starter_rs: str          # filename in kernels/ — the editable starter
    naive_rs: str            # filename in kernels/ — slow baseline (gradient smoke)
    target: float            # throughput at reward 1.0 (GFLOPS for gemm, GB/s for reduction)
    metric: str              # human label for the throughput unit
    train_problem: dict      # problem dims for the train split
    heldout_problem: dict    # different shape -> a meaningful generalization gap
    prompt: str
    cheatsheet: str = field(default="")


# ── cheatsheets injected into the workdir on reset (cuda-oxide is new; no model has
# training data, so the API surface must be handed to the agent every time) ───────
_GEMM_CHEATSHEET = """# cuda-oxide kernel API (inject — the compiler is new, no model has seen it)

Edit ONLY the kernel module in kernel.rs. CONTRACT: keep this exact signature —
the fixed host launches it on a 16x16 grid:

    #[cuda_module]
    mod kernels {
        use super::*;
        #[kernel]
        pub fn sgemm_tiled(m: u32, n: u32, k: u32, alpha: f32,
                           a: &[f32], b: &[f32], beta: f32,
                           mut c: DisjointSlice<f32, thread::Runtime2DIndex>) { ... }
    }

Device APIs: thread::threadIdx_x/y(), thread::blockIdx_x/y(), thread::sync_threads(),
thread::index_2d_runtime(n) -> Option<idx>; `static mut T: SharedArray<f32, N> =
SharedArray::UNINIT;` for shared memory (all access is `unsafe`); `&[T]` read-only
inputs, `DisjointSlice<T>` outputs (.get_mut(idx)).

Device code is no_std: NO Vec / String / Box / dyn / iterators-that-allocate /
format!. Use fixed arrays, slices, while-loops. Optimize the BODY (tiling, register
blocking, unrolling, vectorized loads) — do not change the signature.
"""

_GEMM_PROMPT = (
    "Optimize the cuda-oxide GEMM kernel in kernel.rs to maximize throughput "
    "(GFLOP/s) while staying numerically correct. Keep the "
    "`sgemm_tiled(m,n,k,alpha,a,b,beta,c)` signature (the host launches it on a "
    "16x16 grid). Read CHEATSHEET.md — cuda-oxide is brand new, so rely on it, "
    "not prior CUDA knowledge."
)

_RED_CHEATSHEET = """# cuda-oxide kernel API — parallel reduction (inject; the compiler is new, no model has seen it)

Edit ONLY the kernel module in kernel.rs. CONTRACT: keep this exact signature —
the host launches it on a FIXED 1024 blocks × 256 threads:

    #[cuda_module]
    mod kernels {
        use super::*;
        #[kernel]
        pub fn reduce_sum(n: u32, data: &[f32], mut out: DisjointSlice<f32>) { ... }
    }

`out` has exactly 1024 slots (one partial per block); the host sums them on the CPU
and checks correctness (relative error vs f64 reference < 1e-3). Write block bid's
partial to out[bid]. Launch dims are FIXED — use constants (blockDim.x = 256,
gridDim.x = 1024); do not rely on dynamic blockDim/gridDim.

Device APIs (CONFIRMED to compile on this toolchain):
  thread::threadIdx_x() -> u32, thread::blockIdx_x() -> u32, thread::sync_threads();
  warp::shuffle_down_f32(val: f32, delta: u32) -> f32  (__shfl_down_sync; intra-warp, no barrier);
  `static mut SDATA: SharedArray<f32, 256> = SharedArray::UNINIT;` — shared mem (all access unsafe);
  `data: &[f32]` read-only global input (data[i]); `out.get_unchecked_mut(idx)` (unsafe) to write.

Device code is no_std: NO Vec / String / Box / format! / collecting iterators. Use
SharedArray, raw slices, while-loops, usize arithmetic.

MEMORY-BOUND: bandwidth (GB/s) is dominated by the global load. The win is a COALESCED
grid-stride load (consecutive threads read consecutive addresses). Headroom toward peak:
vectorized float4 loads (4 consecutive f32 per thread), more elements in flight, full
warp-shuffle final stage. Keep the load coalesced.
"""

_RED_PROMPT = (
    "Optimize the cuda-oxide parallel reduction kernel in kernel.rs to maximize "
    "effective memory bandwidth (GB/s) while staying numerically correct (relative "
    "error vs CPU f64 reference < 1e-3). Keep the `reduce_sum(n, data, out)` signature "
    "(the host launches it on a fixed 1024 blocks × 256 threads and sums the 1024 "
    "partials on CPU). Read CHEATSHEET.md — cuda-oxide is brand new, so rely on it, "
    "not prior CUDA knowledge."
)

_NAIVE_GEMM_CHEATSHEET = """# cuda-oxide kernel API — GEMM (inject; the compiler is new, no model has seen it)

You start from NVIDIA's NAIVE sgemm: every thread reads a full row of A and a full
column of B from global memory (zero reuse) -> bandwidth-bound, ~2000 GFLOP/s. Edit
ONLY the kernel module in kernel.rs. CONTRACT: keep this exact signature — the host
launches it on a 16x16 grid:

    #[cuda_module]
    mod kernels {
        use super::*;
        #[kernel]
        pub fn sgemm_naive(m: u32, n: u32, k: u32, alpha: f32,
                           a: &[f32], b: &[f32], beta: f32,
                           mut c: DisjointSlice<f32, thread::Runtime2DIndex>) { ... }
    }

Device APIs: thread::threadIdx_x/y(), thread::blockIdx_x/y(), thread::sync_threads(),
thread::index_2d_runtime(n) -> Option<idx>; `static mut T: SharedArray<f32, N> =
SharedArray::UNINIT;` for shared memory (all access unsafe); `&[T]` read-only inputs,
DisjointSlice<T> outputs (.get_mut(idx)).

Device code is no_std: NO Vec / String / Box / dyn / format! / allocating iterators.
Use fixed arrays, slices, while-loops. The win: shared-memory TILING (cache a tile of A
and B in SharedArray, sync_threads, accumulate) eliminates redundant global reads — the
path from ~2k to >10k GFLOP/s. Then register blocking / vectorized loads. Keep the
sgemm_naive signature and the 16x16 launch.
"""

_NAIVE_GEMM_PROMPT = (
    "Optimize the cuda-oxide GEMM kernel in kernel.rs — it is currently NVIDIA's NAIVE "
    "sgemm (no data reuse, ~2000 GFLOP/s). Add shared-memory tiling (and beyond) to "
    "maximize throughput (GFLOP/s) while staying numerically correct. Keep the "
    "`sgemm_naive(m,n,k,alpha,a,b,beta,c)` signature (the host launches it on a 16x16 "
    "grid). Read CHEATSHEET.md — cuda-oxide is brand new, so rely on it, not prior CUDA "
    "knowledge."
)

_SPECS: dict[str, KernelSpec] = {
    "gemm": KernelSpec(
        name="gemm",
        starter_rs="sgemm_tiled.rs",
        naive_rs="sgemm_naive.rs",
        target=15000.0,
        metric="GFLOPS",
        train_problem={"M": 1024, "N": 1024, "K": 1024},
        heldout_problem={"M": 1536, "N": 1536, "K": 1536},
        prompt=_GEMM_PROMPT,
        cheatsheet=_GEMM_CHEATSHEET,
    ),
    "reduction": KernelSpec(
        name="reduction",
        starter_rs="reduce_opt.rs",
        naive_rs="reduce_naive.rs",
        target=3000.0,
        metric="GB/s",
        train_problem={"N": 33554432},
        heldout_problem={"N": 67108864},
        prompt=_RED_PROMPT,
        cheatsheet=_RED_CHEATSHEET,
    ),
    "naive_gemm": KernelSpec(
        name="naive_gemm",
        starter_rs="sgemm_naive_gemm.rs",
        naive_rs="sgemm_naive_gemm.rs",
        target=12000.0,
        metric="GFLOPS",
        train_problem={"M": 1024, "N": 1024, "K": 1024},
        heldout_problem={"M": 1536, "N": 1536, "K": 1536},
        prompt=_NAIVE_GEMM_PROMPT,
        cheatsheet=_NAIVE_GEMM_CHEATSHEET,
    ),
}


class CudaOxideBackend:
    """GPUBackend: ship a Rust kernel to the deployed ar2-cudaoxide app, return
    {gflops, correct, seconds}. run(kernel_path, problem, kernel_name) — the
    kernel_name selects the server-side grader."""

    def run(self, kernel_path: Path, problem: dict, kernel_name: str = "gemm") -> dict:
        import modal
        src = Path(kernel_path).read_text()
        fn = modal.Function.from_name(_APP, "compile_and_run")
        return fn.remote(src, problem, kernel_name)


class CudaOxideEnv:
    """Optimize a cuda-oxide (Rust->PTX) kernel; reward = clip(throughput/target,0,1)
    gated on correctness."""

    id: str
    split: Split

    def __init__(
        self,
        kernel_name: str = "gemm",
        split: Split = "train",
        problem: dict | None = None,
        target: float | None = None,
        runner=None,
    ) -> None:
        spec = _SPECS[kernel_name]
        self.kernel_name = kernel_name
        self.split = split
        self.problem = dict(problem) if problem else dict(
            spec.train_problem if split == "train" else spec.heldout_problem
        )
        self.target = target if target is not None else spec.target
        shape = "x".join(str(v) for v in self.problem.values())
        self.id = f"cudaoxide-{kernel_name}-{split}-{shape}"
        self._runner = runner or CudaOxideBackend().run
        self._workdir: Path | None = None

    def reset(self) -> TaskSpec:
        spec = _SPECS[self.kernel_name]
        wd = Path(tempfile.mkdtemp(prefix=f"cudaoxide_{self.kernel_name}_"))
        (wd / "kernel.rs").write_text((_KERNELS_DIR / spec.starter_rs).read_text())
        (wd / "CHEATSHEET.md").write_text(spec.cheatsheet)
        self._workdir = wd
        return TaskSpec(
            env_id=self.id,
            split=self.split,
            prompt=spec.prompt,
            workdir=wd,
            payload={**self.problem, "kernel_name": self.kernel_name,
                     "editable_file": "kernel.rs"},
        )

    def score(self, sub: Submission) -> StepResult:
        kpath = sub.workdir / "kernel.rs"
        if not kpath.exists():
            return StepResult(reward=0.0, raw={"error": "kernel.rs missing"},
                              feedback="kernel.rs not found", done=False)
        m = self._runner(kpath, self.problem, self.kernel_name)
        correct = bool(m.get("correct", False))
        gflops = float(m.get("gflops", 0.0))
        reward = min(max(gflops / self.target, 0.0), 1.0) if correct else 0.0
        unit = _SPECS[self.kernel_name].metric
        return StepResult(
            reward=reward,
            raw={"gflops": gflops, "correct": correct, "seconds": m.get("seconds", 0.0)},
            feedback=(f"correct={correct} throughput={gflops:.1f}{unit} "
                      f"target={self.target:.0f} reward={reward:.4f}"),
            done=False,  # kernel opt has no terminal state — iterate until budget
        )


def cuda_oxide_pools(kernels: list[str] | None = None):
    """train/heldout pools across the registered kernels (default: all). The outer
    loop hill-climbs each; AR² then improves the cross-kernel climbing strategy."""
    if kernels is None:
        sel = os.environ.get("AR2_CUDA_OXIDE_KERNELS", "").strip()
        kernels = [k for k in sel.split(",") if k] or None
    names = kernels or list(_SPECS)
    train = [CudaOxideEnv(kernel_name=k, split="train") for k in names]
    heldout = [CudaOxideEnv(kernel_name=k, split="heldout") for k in names]
    return train, heldout


if __name__ == "__main__":
    # env-level modal smoke: reset -> score the starter kernel via the deployed app.
    import sys

    from dotenv import load_dotenv
    load_dotenv("/Users/davidbai/Desktop/autoresearch-hack/.env")
    kname = sys.argv[1] if len(sys.argv) > 1 else "gemm"
    env = CudaOxideEnv(kernel_name=kname, split="train")
    task = env.reset()
    print("reset ->", task.env_id, "workdir:", task.workdir)
    res = env.score(Submission(workdir=task.workdir))
    print("score ->", res.feedback, "| raw:", res.raw)
