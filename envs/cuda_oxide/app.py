"""Persistent Modal app: the H100 grader for the cuda-oxide (Rust->PTX) kernel envs.

DEPLOY ONCE (no ephemeral litter):
    cd ~/Desktop/autoresearch-hack
    export MODAL_PROFILE=... MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...   # from .env
    .venv/bin/modal deploy envs/cuda_oxide/app.py

Then invoke the DEPLOYED functions via modal.Function.from_name(
"ar2-cudaoxide", "<fn>").remote(...).  No top-level dotenv import — the container
image has no python-dotenv, and auth is a host-side concern at deploy/invoke time.

compile_and_run(kernel_src, problem, kernel_name) is the GPU-eval backend: it splices
the agent's kernel into the FIXED host harness for `kernel_name` (timing + correctness
oracle — the integrity boundary the agent never sees) and returns {gflops, correct}.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

import modal

LLVM = "21"
NIGHTLY = "nightly-2026-04-03"
CARGO = "/root/.cargo/bin"

app = modal.App("ar2-cudaoxide")

# Image MUST match the spike's (post-LLVM-fix) byte-for-byte to reuse cached layers.
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.2-devel-ubuntu24.04", add_python="3.11")
    .apt_install(
        "wget", "gnupg", "git", "curl", "build-essential", "ca-certificates",
        "software-properties-common", "lsb-release", "pkg-config", "libssl-dev",
    )
    .run_commands(
        "wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh && chmod +x /tmp/llvm.sh && DEBIAN_FRONTEND=noninteractive /tmp/llvm.sh 21",
        "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends clang-21 llvm-21 llvm-21-dev",
    )
    .run_commands(
        f"curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain {NIGHTLY}",
        f"{CARGO}/rustup component add rust-src rustc-dev --toolchain {NIGHTLY}",
    )
    .run_commands(
        f"{CARGO}/cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide",
        "git clone --depth 1 https://github.com/NVlabs/cuda-oxide.git /opt/cuda-oxide",
    )
    .env({
        "PATH": f"{CARGO}:/usr/lib/llvm-{LLVM}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "CUDA_OXIDE_LLC": f"/usr/lib/llvm-{LLVM}/bin/llc",
    })
)

_REPO = "/opt/cuda-oxide"
_EXAMPLES = "crates/rustc-codegen-cuda/examples"


def _run(cmd, cwd=None, timeout=600) -> dict:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       shell=isinstance(cmd, str), timeout=timeout)
    return {"rc": r.returncode, "out": (r.stdout + r.stderr)[-3000:]}


@app.function(image=image, gpu="H100", timeout=900)
def doctor() -> dict:
    """Confirm the toolchain stands up on a real GPU."""
    return {
        "gpu": _run("nvidia-smi -L"),
        "llc": _run("llc --version | head -4"),
        "rustc": _run("rustc --version"),
        "doctor": _run(["cargo", "oxide", "doctor"], cwd=_REPO),
    }


@app.function(image=image, gpu="H100", timeout=900)
def run_example(name: str) -> dict:
    """Build + run one of cuda-oxide's bundled example kernels on the GPU."""
    return _run(["cargo", "oxide", "run", name], cwd=_REPO)


@app.function(image=image, timeout=120)  # no GPU — just reads the repo
def read_source(rel_path: str) -> str:
    """Return verbatim source of a file in the cuda-oxide repo (for harness templating)."""
    from pathlib import Path
    try:
        return (Path(_REPO) / rel_path).read_text()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


# ── Fixed graders: per-kernel host harnesses the agent's kernel is spliced INTO ──
# Integrity boundary: the agent edits ONLY the kernel module; timing + correctness
# live here and are never shipped to the agent. Each kernel reuses a bundled example
# dir (for its Cargo deps) and substitutes problem dims (__M__/__N__/__K__ ...).
@dataclass(frozen=True)
class _Kernel:
    example: str       # bundled example dir name (also `cargo oxide run <example>`)
    fixed_top: str
    fixed_host: str


_GEMM_TOP = """#![allow(clippy::too_many_arguments)]
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;
use std::time::Instant;
"""

_GEMM_HOST = """
const M: usize = __M__;
const N: usize = __N__;
const K: usize = __K__;
const ALPHA: f32 = 1.0;
const BETA: f32 = 0.0;

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let mut a = vec![0.0f32; M * K];
    let mut b = vec![0.0f32; K * N];
    let c = vec![0.0f32; M * N];
    for i in 0..M { for j in 0..K { a[i * K + j] = ((i + j) % 10) as f32 * 0.1; } }
    for i in 0..K { for j in 0..N { b[i * N + j] = ((i * j) % 10) as f32 * 0.1; } }
    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut c_dev = DeviceBuffer::from_host(&stream, &c).unwrap();
    let module = ctx.load_module_from_file("tiled_gemm.ptx").expect("ptx");
    let module = kernels::from_module(module).expect("typed module");
    let block_size = 16u32;
    let cfg = LaunchConfig {
        grid_dim: ((N as u32).div_ceil(block_size), (M as u32).div_ceil(block_size), 1),
        block_dim: (block_size, block_size, 1),
        shared_mem_bytes: 0,
    };
    let (m_arg, n_arg, k_arg) = (M as u32, N as u32, K as u32);
    let launch = |c_dev: &mut DeviceBuffer<f32>| {
        module.sgemm_tiled((stream).as_ref(), cfg, m_arg, n_arg, k_arg,
                           ALPHA, &a_dev, &b_dev, BETA, c_dev).unwrap();
    };
    launch(&mut c_dev);
    stream.synchronize().unwrap();
    const NUM_RUNS: u32 = 10;
    let start = Instant::now();
    for _ in 0..NUM_RUNS { launch(&mut c_dev); }
    stream.synchronize().unwrap();
    let avg_ms = start.elapsed().as_secs_f64() * 1000.0 / NUM_RUNS as f64;
    let gflops = 2.0 * M as f64 * N as f64 * K as f64 / (avg_ms / 1000.0) / 1e9;
    println!("Average time: {:.3} ms", avg_ms);
    println!("Throughput:   {:.2} GFLOPS", gflops);
    let c_result = c_dev.to_host_vec(&stream).unwrap();
    let mut max_error = 0.0f32;
    for sample in 0..100 {
        let idx = sample * M * N / 100;
        let (row, col) = (idx / N, idx % N);
        let mut expected = 0.0f32;
        for kk in 0..K { expected += a[row * K + kk] * b[kk * N + col]; }
        expected = ALPHA * expected + BETA * c[idx];
        let error = (c_result[idx] - expected).abs();
        if error > max_error { max_error = error; }
    }
    println!("Max error: {:.6e}", max_error);
    if max_error < 1e-3 { println!("SUCCESS"); } else { println!("FAILED"); std::process::exit(1); }
}
"""


_RED_TOP = """#![allow(unused_imports)]
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp};
use cuda_host::cuda_module;
use std::time::Instant;
"""

_RED_HOST = """
const N: usize = __N__;
const NUM_BLOCKS: usize = 1024;
const NUM_THREADS: usize = 256;

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let data: Vec<f32> = (0..N).map(|i| (i % 64) as f32 * 0.01 + 1.0).collect();
    let cpu_sum_f64: f64 = data.iter().map(|&x| x as f64).sum();
    let data_dev = DeviceBuffer::from_host(&stream, &data).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, NUM_BLOCKS).unwrap();
    let module = ctx.load_module_from_file("warp_reduce.ptx").expect("ptx");
    let module = kernels::from_module(module).expect("typed module");
    let cfg = LaunchConfig {
        grid_dim: (NUM_BLOCKS as u32, 1, 1),
        block_dim: (NUM_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let launch = |out_dev: &mut DeviceBuffer<f32>| {
        module.reduce_sum((stream).as_ref(), cfg, N as u32, &data_dev, out_dev).unwrap();
    };
    launch(&mut out_dev);
    stream.synchronize().unwrap();
    const NUM_RUNS: u32 = 10;
    let start = Instant::now();
    for _ in 0..NUM_RUNS { launch(&mut out_dev); }
    stream.synchronize().unwrap();
    let avg_ms = start.elapsed().as_secs_f64() * 1000.0 / NUM_RUNS as f64;
    let bw_gbs = (N as f64 * 4.0) / (avg_ms / 1000.0) / 1e9;
    println!("Average time: {:.3} ms", avg_ms);
    println!("Throughput:   {:.2} GB/s", bw_gbs);
    let partials = out_dev.to_host_vec(&stream).unwrap();
    let gpu_sum: f64 = partials.iter().map(|&x| x as f64).sum();
    let rel_err = (gpu_sum - cpu_sum_f64).abs() / cpu_sum_f64.abs().max(1e-12);
    println!("Max error: {:.6e}", rel_err);
    if rel_err < 1e-3 { println!("SUCCESS"); } else { println!("FAILED"); std::process::exit(1); }
}
"""


_NAIVE_GEMM_TOP = """#![allow(clippy::too_many_arguments)]
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::time::Instant;
"""

_NAIVE_GEMM_HOST = """
const M: usize = __M__;
const N: usize = __N__;
const K: usize = __K__;
const ALPHA: f32 = 1.0;
const BETA: f32 = 0.0;

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let mut a = vec![0.0f32; M * K];
    let mut b = vec![0.0f32; K * N];
    let c = vec![0.0f32; M * N];
    for i in 0..M { for j in 0..K { a[i * K + j] = ((i + j) % 10) as f32 * 0.1; } }
    for i in 0..K { for j in 0..N { b[i * N + j] = ((i * j) % 10) as f32 * 0.1; } }
    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut c_dev = DeviceBuffer::from_host(&stream, &c).unwrap();
    let module = ctx.load_module_from_file("gemm.ptx").expect("ptx");
    let module = kernels::from_module(module).expect("typed module");
    let block_size = 16u32;
    let cfg = LaunchConfig {
        grid_dim: ((N as u32).div_ceil(block_size), (M as u32).div_ceil(block_size), 1),
        block_dim: (block_size, block_size, 1),
        shared_mem_bytes: 0,
    };
    let (m_arg, n_arg, k_arg) = (M as u32, N as u32, K as u32);
    let launch = |c_dev: &mut DeviceBuffer<f32>| {
        module.sgemm_naive((stream).as_ref(), cfg,
                           m_arg, n_arg, k_arg,
                           ALPHA, &a_dev, &b_dev, BETA, c_dev).unwrap();
    };
    launch(&mut c_dev);
    stream.synchronize().unwrap();
    const NUM_RUNS: u32 = 10;
    let start = Instant::now();
    for _ in 0..NUM_RUNS { launch(&mut c_dev); }
    stream.synchronize().unwrap();
    let avg_ms = start.elapsed().as_secs_f64() * 1000.0 / NUM_RUNS as f64;
    let gflops = 2.0 * M as f64 * N as f64 * K as f64 / (avg_ms / 1000.0) / 1e9;
    println!("Average time: {:.3} ms", avg_ms);
    println!("Throughput:   {:.2} GFLOPS", gflops);
    let c_result = c_dev.to_host_vec(&stream).unwrap();
    let mut max_error = 0.0f32;
    for sample in 0..100 {
        let idx = sample * M * N / 100;
        let (row, col) = (idx / N, idx % N);
        let mut expected = 0.0f32;
        for kk in 0..K { expected += a[row * K + kk] * b[kk * N + col]; }
        expected = ALPHA * expected + BETA * c[idx];
        let error = (c_result[idx] - expected).abs();
        if error > max_error { max_error = error; }
    }
    println!("Max error: {:.6e}", max_error);
    if max_error < 1e-3 { println!("SUCCESS"); } else { println!("FAILED"); std::process::exit(1); }
}
"""


_TMA_COPY_TOP = """#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::missing_safety_doc)]
use cuda_core::{
    CudaContext, DeviceBuffer, LaunchConfig,
    sys::{
        self as cuda_sys, CUtensorMap,
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE, cuTensorMapEncodeTiled,
    },
};
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_expect_tx,
    mbarrier_init, mbarrier_try_wait,
};
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;
use std::mem::MaybeUninit;
use std::time::Instant;
"""

_TMA_COPY_HOST = """
const TILE_W: u32  = __TILE_W__;
const TILE_H: u32  = __TILE_H__;
const TILES_X: i32 = __TILES_X__;
const TILES_Y: i32 = __TILES_Y__;

const TILE_SIZE: usize  = (TILE_W * TILE_H) as usize;
const NUM_TILES: usize  = (TILES_X * TILES_Y) as usize;
const TENSOR_W: u64     = TILE_W as u64 * TILES_X as u64;
const TENSOR_H: u64     = TILE_H as u64 * TILES_Y as u64;
const TENSOR_SIZE: usize = (TENSOR_W * TENSOR_H) as usize;
const TOTAL_BYTES: usize = TENSOR_SIZE * 4;

fn create_tma_descriptor(
    global_address: *mut std::ffi::c_void,
    width: u64,
    height: u64,
    tile_width: u32,
    tile_height: u32,
) -> CUtensorMap {
    let mut tensor_map = MaybeUninit::<CUtensorMap>::uninit();
    let tensor_rank = 2u32;
    let global_dim: [u64; 2] = [width, height];
    let global_strides: [u64; 1] = [width * std::mem::size_of::<f32>() as u64];
    let box_dim: [u32; 2] = [tile_width, tile_height];
    let element_strides: [u32; 2] = [1, 1];
    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            tensor_rank,
            global_address,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if result != cuda_sys::cudaError_enum_CUDA_SUCCESS {
        eprintln!("cuTensorMapEncodeTiled failed: {:?}", result);
        std::process::exit(1);
    }
    unsafe { tensor_map.assume_init() }
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let host_input: Vec<f32> = (0..TENSOR_SIZE).map(|i| i as f32).collect();
    let dev_tensor = DeviceBuffer::from_host(&stream, &host_input).unwrap();
    let mut dev_output = DeviceBuffer::<f32>::zeroed(&stream, NUM_TILES * TILE_SIZE).unwrap();
    let ptr = dev_tensor.cu_deviceptr();
    let tensor_map = create_tma_descriptor(
        ptr as *mut std::ffi::c_void,
        TENSOR_W, TENSOR_H, TILE_W, TILE_H,
    );
    let dev_tensor_map = DeviceBuffer::from_host(&stream, &tensor_map.opaque[..]).unwrap();
    let tensor_map_ptr = dev_tensor_map.cu_deviceptr() as *const TmaDescriptor;
    let module = ctx.load_module_from_file("tma_copy.ptx").expect("ptx");
    let module = kernels::from_module(module).expect("typed module");
    let cfg = LaunchConfig {
        grid_dim: (NUM_TILES as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let launch = |out: &mut DeviceBuffer<f32>| {
        module.tma_copy_bench((stream).as_ref(), cfg,
                              tensor_map_ptr, TILES_X, TILES_Y, out).unwrap();
    };
    launch(&mut dev_output);
    stream.synchronize().unwrap();
    const NUM_RUNS: u32 = 10;
    let start = Instant::now();
    for _ in 0..NUM_RUNS { launch(&mut dev_output); }
    stream.synchronize().unwrap();
    let avg_ms = start.elapsed().as_secs_f64() * 1000.0 / NUM_RUNS as f64;
    let bw_gbs = TOTAL_BYTES as f64 / (avg_ms / 1000.0) / 1e9;
    println!("Average time: {:.3} ms", avg_ms);
    println!("Throughput:   {:.2} GB/s", bw_gbs);
    let host_output = dev_output.to_host_vec(&stream).unwrap();
    let mut max_error = 0.0f32;
    for tile_idx in 0..NUM_TILES {
        let tx = tile_idx % TILES_X as usize;
        let ty = tile_idx / TILES_X as usize;
        for elem in 0..TILE_SIZE {
            let local_row = elem / TILE_W as usize;
            let local_col = elem % TILE_W as usize;
            let global_row = ty * TILE_H as usize + local_row;
            let global_col = tx * TILE_W as usize + local_col;
            let expected = (global_row * TENSOR_W as usize + global_col) as f32;
            let got = host_output[tile_idx * TILE_SIZE + elem];
            let err = (got - expected).abs();
            if err > max_error { max_error = err; }
        }
    }
    println!("Max error: {:.6e}", max_error);
    if max_error < 0.5 { println!("SUCCESS"); } else { println!("FAILED"); std::process::exit(1); }
}
"""


_KERNELS: dict[str, _Kernel] = {
    "gemm": _Kernel("tiled_gemm", _GEMM_TOP, _GEMM_HOST),
    "reduction": _Kernel("warp_reduce", _RED_TOP, _RED_HOST),
    "naive_gemm": _Kernel("gemm", _NAIVE_GEMM_TOP, _NAIVE_GEMM_HOST),
    "tma_copy": _Kernel("tma_copy", _TMA_COPY_TOP, _TMA_COPY_HOST),
}


@app.function(image=image, gpu="H100", timeout=900)
def compile_and_run(kernel_src: str, problem: dict | None = None,
                    kernel_name: str = "gemm") -> dict:
    """Splice the agent's kernel into the fixed host harness for `kernel_name`, build
    via cargo-oxide, run on the GPU, return {gflops, correct, seconds}. The agent
    never sees the host/timing/correctness code (the integrity boundary)."""
    import re
    import subprocess
    from pathlib import Path

    k = _KERNELS[kernel_name]
    host = k.fixed_host
    for key, val in (problem or {}).items():
        host = host.replace(f"__{key}__", str(int(val)))
    main_rs = k.fixed_top + "\n" + kernel_src.strip() + "\n" + host

    proj = Path(_REPO) / _EXAMPLES / k.example
    (proj / "src" / "main.rs").write_text(main_rs)  # overwrite the example's kernel + host
    try:
        # Invoke exactly like run_example (proven): from the repo root, by example name.
        r = subprocess.run(["cargo", "oxide", "run", k.example], cwd=_REPO,
                           capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"gflops": 0.0, "correct": False, "seconds": 0.0, "error": "timeout"}

    out = r.stdout + "\n" + r.stderr
    if r.returncode != 0:
        return {"gflops": 0.0, "correct": False, "seconds": 0.0, "error": out[-1800:]}
    correct = "SUCCESS" in out
    gm = re.search(r"Throughput:\s*([\d.]+)", out)
    tm = re.search(r"Average time:\s*([\d.]+)\s*ms", out)
    gflops = float(gm.group(1)) if gm else 0.0
    seconds = (float(tm.group(1)) / 1000.0) if tm else 0.0
    return {
        "gflops": gflops if correct else 0.0,
        "correct": correct,
        "seconds": seconds,
        "tail": out[-400:],
    }
