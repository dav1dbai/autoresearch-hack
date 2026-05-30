# cuda-oxide Derisk Brief

**Repo**: https://github.com/NVlabs/cuda-oxide (released 2026-05-09, alpha)
**Researched from**: shallow clone at `/tmp/cuda-oxide`, commit depth 1

---

## A. Kernel Cheatsheet

cuda-oxide compiles pure Rust to PTX via a custom rustc backend. Host and device code live in **one file**, built with `cargo oxide run`. No separate device crate, no `cfg(cuda_device)` splits.

### Minimal imports

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;  // or: use cuda_device::cuda_module;
```

### Kernel anatomy

```rust
#[cuda_module]
mod kernels {
    use super::*;

    // #[kernel] marks the GPU entry point. Must return ().
    // DisjointSlice<T> = safe mutable output slice (one write per thread).
    // &[T] = read-only input slice.
    #[kernel]
    pub fn my_kernel(input: &[f32], scale: f32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();          // blockIdx.x * blockDim.x + threadIdx.x
        if let Some(elem) = out.get_mut(idx) { // bounds-safe; None for out-of-range threads
            *elem = input[idx.get()] * scale;
        }
    }
}
```

Key device APIs (from `cuda_device`):

| API | Description |
|-----|-------------|
| `thread::index_1d()` | Global 1D thread index (returns `ThreadIndex`) |
| `thread::threadIdx_x/y/z()` | Per-block thread ID, returns `u32` |
| `thread::blockIdx_x/y/z()` | Block ID, returns `u32` |
| `thread::blockDim_x/y/z()` | Block size, returns `u32` |
| `thread::sync_threads()` | `__syncthreads()` barrier |
| `thread::index_2d_row()` / `index_2d_col()` | 2D row/col indices |
| `DisjointSlice<T>.get_mut(idx)` | Bounds-safe parallel write |
| `static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT` | Static shared memory |

Helper functions reachable from `#[kernel]` are auto-compiled for the GPU — no annotation needed. Use `#[device]` only for cross-crate or standalone device functions.

Unsupported in device code: `Vec`, `String`, `Box`, `std` I/O, `dyn Trait`. Use fixed arrays, slices, and generics instead.

### Host launch

```rust
fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let input = DeviceBuffer::from_host(&stream, &host_data).unwrap();
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).unwrap();   // load embedded PTX
    module
        .my_kernel(
            &stream,
            LaunchConfig::for_num_elems(N as u32),  // 256 threads/block, auto grid
            &input,
            2.5f32,
            &mut output,
        )
        .unwrap();

    let result = output.to_host_vec(&stream).unwrap();
}
```

`LaunchConfig::for_num_elems(N)` uses 256 threads/block, `grid = ceil(N/256)`. For custom shapes:

```rust
LaunchConfig {
    grid_dim: (grid_x, grid_y, 1),
    block_dim: (16, 16, 1),
    shared_mem_bytes: 0,
}
```

---

## B. Starter Kernel

The **correct-but-naive** GEMM and the **tiled GEMM with shared memory** both exist verbatim in the repo. Use `tiled_gemm` as the editable starting point — it compiles cleanly on any sm_XX without Blackwell intrinsics.

### Starter: `tiled_gemm` (from `crates/rustc-codegen-cuda/examples/tiled_gemm/src/main.rs`)

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

const TILE_SIZE: usize = 16;

#[cuda_module]
mod kernels {
    use super::*;

    /// Tiled SGEMM: C = alpha * A * B + beta * C
    /// A: M x K row-major, B: K x N row-major, C: M x N row-major
    #[kernel]
    pub fn sgemm_tiled(
        m: u32, n: u32, k: u32,
        alpha: f32, a: &[f32], b: &[f32],
        beta: f32, mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * TILE_SIZE + ty;
        let col = thread::blockIdx_x() as usize * TILE_SIZE + tx;
        let (m_sz, n_sz, k_sz) = (m as usize, n as usize, k as usize);

        let mut sum = 0.0f32;
        let mut tile = 0usize;
        while tile < k_sz.div_ceil(TILE_SIZE) {
            let tile_start = tile * TILE_SIZE;
            let smem_idx = ty * TILE_SIZE + tx;
            unsafe {
                TILE_A[smem_idx] = if row < m_sz && tile_start + tx < k_sz {
                    a[row * k_sz + tile_start + tx]
                } else { 0.0 };
                TILE_B[smem_idx] = if tile_start + ty < k_sz && col < n_sz {
                    b[(tile_start + ty) * n_sz + col]
                } else { 0.0 };
            }
            thread::sync_threads();
            unsafe {
                let mut i = 0usize;
                while i < TILE_SIZE { sum += TILE_A[ty * TILE_SIZE + i] * TILE_B[i * TILE_SIZE + tx]; i += 1; }
            }
            thread::sync_threads();
            tile += 1;
        }

        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n_sz) } {
            if row < m_sz {
                if let Some(c_elem) = c.get_mut(c_idx) {
                    *c_elem = alpha * sum + beta * (*c_elem);
                }
            }
        }
    }
}

fn main() {
    // ... (see tiled_gemm/src/main.rs for full host code with timing)
    let block_size = 16u32;
    let cfg = LaunchConfig {
        grid_dim: ((N as u32).div_ceil(block_size), (M as u32).div_ceil(block_size), 1),
        block_dim: (block_size, block_size, 1),
        shared_mem_bytes: 0,
    };
    // module.sgemm_tiled(&stream, cfg, M as u32, N as u32, K as u32, 1.0, &a_dev, &b_dev, 0.0, &mut c_dev)
}
```

### What `gemm_sol` does that's Blackwell-only

`gemm_sol` (`crates/rustc-codegen-cuda/examples/gemm_sol/src/main.rs`, 267 KB) uses:

- **tcgen05** — 5th-gen tensor cores (`tcgen05_mma_f16`, `tcgen05_alloc`, `TmemGuard`): SM 100+ only
- **TMA** (`cp_async_bulk_tensor_2d_g2s`) — Tensor Memory Accelerator: SM 90+ (Hopper+), required for the fast path
- **WGMMA** — Warpgroup MMA, SM 90 only
- **Cluster Launch Control (CLC)** — `clc_try_cancel`, SM 100+ only
- **`cta_group::2`** — Pair-UMMA, SM 100+ only
- **mbarrier** — `mbarrier_init`/`mbarrier_try_wait_parity`: SM 90+

The 8-kernel progression goes from naive tiled (SM 75+) through Phase 4D at 868 TFLOPS (SM 100 required). Only Phases 1 and 1.5 might theoretically compile on Hopper (SM 90) but still use TMA, which requires at least Hopper. The naive `gemm` example (`examples/gemm/src/main.rs`) and `tiled_gemm` are the only portable starting points.

---

## C. Eval Mechanics

### Build and run commands

```bash
# Inside the cuda-oxide workspace:
cargo oxide run <example_name>

# For a standalone project (outside repo):
cargo oxide run <project_name>

# Build only (no GPU needed — for PTX generation on CPU-only CI):
cargo oxide build <example_name>

# Show full compilation pipeline (MIR → PTX dumped to stdout):
cargo oxide pipeline <example_name>

# Validate environment:
cargo oxide doctor
```

Under the hood, `cargo oxide run` sets:
```
RUSTFLAGS="-Z codegen-backend=<path>/librustc_codegen_cuda.so -C opt-level=3 -C debug-assertions=off -Z mir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0"
```
then runs `cargo run --release` in the example directory. The PTX is generated alongside the binary.

Architecture targeting:
```bash
cargo oxide run --arch sm_90  gemm_sol   # explicit
# or auto-detected from device 0 compute capability
```

### Output format

The `gemm` and `tiled_gemm` examples print:
```
Average time: 12.345 ms
Throughput:   XX.XX GFLOPS
Max error: 1.234e-06
✓ SUCCESS: Tiled GEMM computed correctly!
```

The `gemm_sol` example prints (per phase, per size):
```
[gemm_sol_tiled] 4096x4096x4096: PASSED (max_err=1.5e-03) 182 TFLOPS (12.1% of cublasLt SoL)
```
(Exact format from bench — timing via `cudaEventRecord`/`cudaEventElapsedTime`, 10 warmup + 100 timed iterations.)

### Eval harness design

**Step 1 — Compile agent's kernel:**
```bash
cd /path/to/agent_kernel_project
cargo oxide build 2>&1
# Exit 0 = compile success. Exit nonzero = compile error (report stderr as reward=0)
```

**Step 2 — Verify correctness:**
Run the binary; parse stdout for `✓ SUCCESS` and `Max error: <N>`. Alternatively, the harness can:
- Run the kernel on known inputs
- Compare to a numpy/CPU reference: `np.allclose(result, expected, atol=1e-3)` (f32 precision allows 1e-3 tolerance)

```python
import subprocess, numpy as np

proc = subprocess.run(["./target/release/my_gemm"], capture_output=True, text=True, timeout=60)
correct = "SUCCESS" in proc.stdout and proc.returncode == 0
```

**Step 3 — Measure TFLOPS:**
The existing examples compute TFLOPS inline. For an eval harness, parse stdout:
```python
import re
match = re.search(r"([\d.]+)\s+(?:G|T)FLOPS", proc.stdout)
tflops = float(match.group(1)) * (1e-3 if "GFLOPS" in match.group(0) else 1.0)
```

Or instrument the host code with `cudaEventRecord`/`cudaEventElapsedTime` (as in `gemm_sol`):
```rust
// FLOP formula for GEMM:
let flops = 2.0 * M as f64 * N as f64 * K as f64;  // multiply-add = 2 ops
let tflops = flops / (elapsed_ms / 1000.0) / 1e12;
```

**GEMM FLOP formula:**
```
FLOPs = 2 * M * N * K
```
(Each output element requires K multiply-adds = 2K FLOPs. Total = M*N*2K.)

**Step 4 — Compute reward:**
```python
# SoL reference: cublasLtMatmul on B200 (FP16 in, FP32 compute, TN format)
# Per gemm_sol/bench/README.md and gemm_sol/README.md:
cublas_sol = {4096: 1502, 8192: 1402, 16384: 1526}  # TFLOPS by matrix size

reward = achieved_tflops / cublas_sol[matrix_size]  # 0.0 to ~0.58 (Phase 4D best)
```

**Practical SoL reference for B200 (sm_100, 148 SMs):**
- Peak FP16 tensor core throughput: ~2250 TFLOPS (theoretical hardware max)
- cublasLtMatmul (practical ceiling): ~1400-1526 TFLOPS at 4K-16K sizes
- gemm_sol Phase 4D best: 868 TFLOPS = 57.8% of cublasLt (at 4K)

For the inner-loop climb starting from `tiled_gemm` (16x16 tiles, ~GFLOPS range), a more useful reward baseline is percentage of a Phase 1 reference (~190 TFLOPS) rather than cuBLAS ceiling, to give the LLM a reward signal from the start.

---

## D. LLM Feasibility

### What an LLM can do well

1. **Simple tiling + shared memory** (`tiled_gemm` level): The pattern is in the LLM's training data (CUDA C++ tiled GEMM is classic). cuda-oxide's Rust syntax is close enough that the mapping is mechanical. A model can iterate tile size, block size, unroll loop hints.

2. **Warp reductions, vectorized loads**: Standard CUDA optimizations that map cleanly to cuda-oxide's warp shuffle APIs (`warp::shuffle_xor_f32`, etc.).

3. **Compile-error iteration**: The compiler errors from cuda-oxide are standard rustc errors (forbidden crate use, type mismatches) — clear and actionable.

### Failure surface

1. **Alpha compiler bugs (highest risk)**: The compiler is 3 weeks old. The README explicitly warns: "you should expect bugs, incomplete features, and API breakage." Some MIR patterns cause silent miscompilation or ICE. The `error_*` examples in the repo (`error_drop_glue`, `error_wgmma_mma_unimplemented`, etc.) document known-broken patterns. An LLM may wander into these.

2. **No LLM training data for cuda-oxide**: The library was released 2026-05-09. No models trained before August 2025 have seen any cuda-oxide code. The LLM is flying blind on exact API names (`DisjointSlice`, `thread::index_2d_runtime`, `SharedArray::UNINIT`, `mbarrier_init`, etc.). You **must inject the cheatsheet** (Section A above) into every prompt.

3. **Blackwell-only tensor core APIs**: Anything beyond tiled GEMM (TMA, tcgen05, WGMMA, CLC) requires SM 90+/SM 100+ hardware intrinsics that are not abstractable — you get CUDA PTX assembly in Rust clothing. An LLM that tries to port CUTLASS/Triton optimizations directly will fail because the APIs are entirely new.

4. **Setup complexity**: requires nightly-2026-04-03, LLVM 21+, CUDA 12.8+, clang-21, bindgen. On any non-devcontainer machine this is a 30-min setup. CI on GPU-equipped infra is required. The devcontainer (`.devcontainer/`) handles this.

5. **Unsupported Rust patterns in device code**: `Vec`, iterators that allocate, `format!`, `dyn Trait`, panics with messages — all silently excluded by the collector or trap at runtime. An LLM writing idiomatic Rust will hit these.

6. **Borrow checker on `static mut`**: All shared memory is `static mut SharedArray`. Every access is `unsafe`. The LLM must produce `unsafe` blocks consistently or the code won't compile.

7. **2D indexing ABI**: The `DisjointSlice<f32, thread::Runtime2DIndex>` + `unsafe { thread::index_2d_runtime(n_sz) }` pattern is non-obvious and essential for 2D kernels. Missing this = type error.

### What docs/scaffolding to inject

At minimum, inject:
- Section A of this brief (kernel cheatsheet + host launch pattern)
- The full `tiled_gemm/src/main.rs` source as a working reference
- The `vecadd/src/main.rs` source as a minimal example
- The constraint: "no Vec/String/Box/std in device code; use SharedArray for shared memory; all shared memory access requires unsafe"
- The build command: `cargo oxide run <name>` and what stdout should look like

Also useful: Section B of the `cuda-oxide-book/gpu-programming/kernels-and-device-functions.md` (supported/unsupported feature table).

### Is there enough gradient for a climbing curve?

**Yes, but only up to a ceiling.** The optimization ladder from `tiled_gemm` is real:

| Step | Technique | Expected gain | LLM feasibility |
|------|-----------|---------------|-----------------|
| 0 | Naive (1 thread/elem) | ~5-10 GFLOPS | Baseline |
| 1 | 16x16 shared memory tiling | ~100-300 GFLOPS | High — standard pattern |
| 2 | Larger tiles (32x32), more unrolling | ~300-500 GFLOPS | Medium |
| 3 | Register blocking, vectorized loads | ~500-800 GFLOPS | Medium |
| 4 | Warp-level reductions, prefetching | ~1-5 TFLOPS | Hard |
| 5 | TMA + tcgen05 (Blackwell-only) | 100+ TFLOPS | Requires sm_100, very hard |

Steps 1-3 are accessible to LLMs with injected scaffolding. Steps 4-5 require B200 hardware and Blackwell-specific APIs that no model has training data for. The climbing curve is real through Step 3 but hits a wall before the cuBLAS SoL range.

---

## E. Verdict

**Yellow** — with important caveats.

**Positive signals:**
- A working naive GEMM (`tiled_gemm`) exists in the repo and compiles today. The host/device API is learnable from the cheatsheet. Tiling optimizations through ~1-5 TFLOPS are tractable for a model with scaffolding.
- The compile-error loop is fast: `cargo oxide run` takes ~30s including rustc + PTX. Tight inner loop for iteration.
- The correctness check is built-in: existing examples verify output vs CPU reference and exit nonzero on failure.

**Risk signals:**
- The compiler is 3 weeks old and alpha. Unexpected ICEs or silent miscompilations on unusual Rust patterns are a real risk. An agent that wanders into unsupported MIR patterns may spend many iterations on spurious compile errors.
- Peak performance on B200 requires Blackwell intrinsics (tcgen05, TMA, CLC) that are not accessible to general-purpose LLMs. The achievable ceiling without those is ~1-5 TFLOPS — roughly 0.1-0.3% of cuBLAS SoL. The reward signal exists but the range is narrow if your SoL is 1500 TFLOPS.
- If the goal is a *climbing curve toward cuBLAS SoL* specifically, cuda-oxide on B200 without injecting the full `gemm_sol` progression as a scaffold will plateau far below. If the goal is a climbing curve from naive to "competent tiled GEMM," cuda-oxide works.

**Single biggest risk:** The alpha compiler. An LLM-generated kernel that uses a legal-looking Rust pattern (an `Iterator`, a slightly unusual struct layout, a closure in an unexpected position) may trigger a compiler bug that produces an ICE or a silently wrong PTX. The agent loop will stall trying to debug what is a compiler issue, not a logic issue. Mitigation: restrict the agent to patterns that are demonstrably working (from the 46 existing examples) and enforce this via the scaffolding prompt.

**Alternative if this is too risky:** Triton (Python, no alpha bugs, mature LLM training data, B200 support, cuBLAS-comparable peak). The cuda-oxide value prop is a Rust-native approach to a space where the LLM has zero training data — good for novelty, bad for reliability in an automated inner loop.
