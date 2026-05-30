"""triton-gemm env — the agent optimizes a Triton GEMM kernel toward GFLOPS,
scored on the deployed `ar2-triton` Modal app (H100). Conforms to
`harness.contracts.Env`, so it drops into the outer loop like any other env.

Self-contained on purpose (no imports from the refactoring envs/); can be moved to
envs/triton_gemm.py + added to a pool later. The agent edits ONLY matmul.py; the
fixed host harness (timing + correctness) lives in app.py's triton_run.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from harness.contracts import Env, Split, StepResult, Submission, TaskSpec

_HERE = Path(__file__).parent
_STARTER = _HERE / "kernels" / "matmul_tiled.py"
_APP = "ar2-triton"

# Injected into the workdir on reset — teach the agent the Triton API + the
# anti-shortcut rule (torch.matmul bypass is not accepted).
_CHEATSHEET = """# Triton matmul kernel API

Edit ONLY the kernel in matmul.py. CONTRACT:
  - The file MUST define `matmul(A, B) -> Tensor` that dispatches to an `@triton.jit` kernel.
  - The host harness imports `matmul` directly — do NOT just call `torch.matmul` inside it;
    the server checks for `@triton.jit` and rejects files that lack it.
  - Keep the `matmul(A, B)` signature. Tune BLOCK_M/N/K and the kernel body.

Triton API cheatsheet:

    import triton
    import triton.language as tl

    @triton.jit
    def kernel(ptr, ..., BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(ptr + offs, mask=offs < N)
        tl.store(out + offs, x, mask=offs < N)

    # Launch grid
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    kernel[grid](ptr, ..., BLOCK=64)

Key tl ops: tl.load, tl.store, tl.dot (tensor-core matmul on 2-D tiles),
tl.zeros, tl.arange, tl.program_id, tl.cdiv, tl.constexpr.

tl.dot(a, b): a is (BLOCK_M, BLOCK_K), b is (BLOCK_K, BLOCK_N) — uses tensor cores.
Inputs are fp16; accumulate in fp32 (tl.float32), store back as fp16.

Optimization knobs: BLOCK_M, BLOCK_N, BLOCK_K (must be powers of 2, >=16 for tl.dot);
num_warps (default 4, try 8); num_stages for software pipelining; tl.dot's
allow_tf32 flag; pointer swizzling for L2 cache efficiency.
"""


class TritonBackend:
    """Ship a Triton kernel source to the deployed ar2-triton app, return
    {gflops, correct, seconds}. Same run(kernel_path, problem) shape as
    CudaOxideBackend so the env<->backend wiring is identical."""

    def run(self, kernel_path: Path, problem: dict) -> dict:
        import modal
        src = Path(kernel_path).read_text()
        fn = modal.Function.from_name(_APP, "triton_run")
        return fn.remote(src, problem)


class TritonGemmEnv:
    """Optimize a Triton GEMM kernel; reward = clip(gflops / target_gflops, 0, 1)."""

    id: str
    split: Split

    def __init__(
        self,
        split: Split = "train",
        M: int = 1024,
        N: int = 1024,
        K: int = 1024,
        target_gflops: float = 150000.0,
        runner=None,
    ) -> None:
        self.id = f"triton-gemm-{split}-{M}x{N}x{K}"
        self.split = split
        self.M, self.N, self.K = M, N, K
        self.target_gflops = target_gflops
        self._runner = runner or TritonBackend().run
        self._workdir: Path | None = None

    def reset(self) -> TaskSpec:
        wd = Path(tempfile.mkdtemp(prefix="triton_gemm_"))
        (wd / "matmul.py").write_text(_STARTER.read_text())
        (wd / "CHEATSHEET.md").write_text(_CHEATSHEET)
        self._workdir = wd
        return TaskSpec(
            env_id=self.id,
            split=self.split,
            prompt=(
                "Optimize the Triton matmul kernel in matmul.py to maximize throughput "
                "(GFLOP/s) while staying numerically correct. Keep the `matmul(A, B)` "
                "signature and ensure the file contains an `@triton.jit` kernel that "
                "implements the matmul — the host will reject a bare torch.matmul call. "
                "Read CHEATSHEET.md for the Triton API. Tune BLOCK_M/N/K, num_warps, "
                "num_stages, and the kernel body (tiling, software pipelining, swizzling)."
            ),
            workdir=wd,
            payload={"M": self.M, "N": self.N, "K": self.K, "editable_file": "matmul.py"},
        )

    def score(self, sub: Submission) -> StepResult:
        kpath = sub.workdir / "matmul.py"
        if not kpath.exists():
            return StepResult(reward=0.0, raw={"error": "matmul.py missing"},
                              feedback="matmul.py not found", done=False)
        m = self._runner(kpath, {"M": self.M, "N": self.N, "K": self.K})
        correct = bool(m.get("correct", False))
        gflops = float(m.get("gflops", 0.0))
        reward = min(max(gflops / self.target_gflops, 0.0), 1.0) if correct else 0.0
        return StepResult(
            reward=reward,
            raw={"gflops": gflops, "correct": correct, "seconds": m.get("seconds", 0.0)},
            feedback=(f"correct={correct} gflops={gflops:.1f} "
                      f"target={self.target_gflops:.0f} reward={reward:.4f}"),
            done=False,
        )


# Pools (train/heldout differ by shape so a gap in reward is meaningful)
def triton_gemm_pools(target_gflops: float = 150000.0):
    train = [TritonGemmEnv(split="train", M=1024, N=1024, K=1024, target_gflops=target_gflops)]
    heldout = [TritonGemmEnv(split="heldout", M=1536, N=1536, K=1536, target_gflops=target_gflops)]
    return train, heldout


if __name__ == "__main__":
    # env-level modal smoke: reset -> score the starter kernel via the deployed app.
    from dotenv import load_dotenv
    load_dotenv("/Users/davidbai/Desktop/autoresearch-hack/.env")
    env = TritonGemmEnv(split="train", M=512, N=512, K=512)
    task = env.reset()
    print("reset ->", task.env_id, "workdir:", task.workdir)
    res = env.score(Submission(workdir=task.workdir))
    print("score ->", res.feedback, "| raw:", res.raw)
