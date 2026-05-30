"""Modal entrypoints for Qwen3.6-27B inference benchmarking on H100."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import modal

_MODAL_BENCH = Path("/bench")
BENCH_ROOT = _MODAL_BENCH if _MODAL_BENCH.is_dir() else Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parents[3] if len(BENCH_ROOT.parents) >= 4 else BENCH_ROOT.parent

app = modal.App("qwen36-inference-bench")
results_volume = modal.Volume.from_name("qwen36-bench-results", create_if_missing=True)

secrets = [modal.Secret.from_dotenv()] if (REPO_ROOT / ".env").exists() else []

client_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("httpx>=0.27", "pyyaml>=6", "python-dotenv>=1")
    .add_local_dir(str(BENCH_ROOT), remote_path="/bench")
)

feasibility_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential", "ninja-build")
    .pip_install("httpx>=0.27", "pyyaml>=6", "python-dotenv>=1", "numpy")
    .run_commands(
        "python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
        "git clone --recursive --branch mpk --depth 1 https://github.com/mirage-project/mirage /opt/mirage || true",
        "python -m pip install -e /opt/mirage -v || true",
    )
    .env({"MIRAGE_HOME": "/opt/mirage", "HF_HOME": "/hf"})
    .add_local_dir(str(BENCH_ROOT), remote_path="/bench")
)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl")
    .pip_install("httpx>=0.27", "pyyaml>=6", "python-dotenv>=1", "numpy")
    .run_commands(
        "python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
        "python -m pip install 'vllm>=0.19.0'",
    )
    .env({"HF_HOME": "/hf"})
    .add_local_dir(str(BENCH_ROOT), remote_path="/bench")
)


def _gpu_slug() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0).replace(" ", "_")
    except Exception:
        return "unknown"


@app.function(
    image=client_image,
    timeout=60 * 60,
    secrets=secrets,
    volumes={"/results": results_volume},
)
def run_sweep_remote(
    base_url: str,
    engine: str = "vllm",
    model: str | None = None,
    workload: list[str] | None = None,
) -> dict:
    os.chdir("/bench")
    cmd = [
        sys.executable,
        "run_sweep.py",
        "--base-url",
        base_url,
        "--engine",
        engine,
        "-o",
        f"/results/baselines_{engine}.json",
    ]
    if model:
        cmd.extend(["--model", model])
    if workload:
        for w in workload:
            cmd.extend(["--workload", w])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}


@app.function(
    image=feasibility_image,
    gpu="H100",
    timeout=90 * 60,
    secrets=secrets,
    volumes={"/results": results_volume},
)
def run_feasibility_remote(
    base_url: str | None = None,
    engine: list[str] | None = None,
    smoke_mirage: bool = False,
) -> dict:
    os.chdir("/bench")
    slug = _gpu_slug()
    out = f"/results/feasibility_{slug}.json"
    cmd = [sys.executable, "run_feasibility.py", "-o", out]
    if base_url:
        cmd.extend(["--base-url", base_url])
    if smoke_mirage:
        cmd.append("--smoke-mirage")
    if engine:
        for e in engine:
            cmd.extend(["--engine", e])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    payload = {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
    try:
        payload["results"] = json.loads(Path(out).read_text())
    except Exception:
        pass
    results_volume.commit()
    return payload


def _wait_vllm(port: int = 8000, timeout_s: float = 900.0) -> None:
    import time
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as ex:
            last = str(ex)
        time.sleep(3)
    raise RuntimeError(f"vLLM not ready after {timeout_s}s: {last}")


@app.function(
    image=vllm_image,
    gpu="H100",
    timeout=120 * 60,
    secrets=secrets,
    volumes={"/results": results_volume},
)
def run_vllm_bench_remote(
    model: str = "Qwen/Qwen3.6-27B-FP8",
    max_model_len: int = 8192,
    n_requests: int = 10,
    warmup: int = 2,
) -> dict:
    import signal

    os.chdir("/bench")
    slug = _gpu_slug()
    port = 8000
    server = subprocess.Popen(
        [
            "vllm",
            "serve",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--max-model-len",
            str(max_model_len),
            "--reasoning-parser",
            "qwen3",
            "--language-model-only",
            "--enable-prefix-caching",
            "--gpu-memory-utilization",
            "0.85",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    payload: dict = {"gpu": slug, "model": model, "engine": "vllm"}
    try:
        _wait_vllm(port)
        workloads = ["D_batch1_decode", "B_long_decode"]
        runs = []
        for wl in workloads:
            out = f"/results/baselines_{slug}_vllm_{wl}.json"
            cmd = [
                sys.executable,
                "run_sweep.py",
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--engine",
                "vllm",
                "--model",
                model,
                "--workload",
                wl,
                "--n-requests",
                str(n_requests),
                "--warmup",
                str(warmup),
                "-o",
                out,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            runs.append({"workload": wl, "returncode": proc.returncode, "stdout": proc.stdout[-2000:]})
            if Path(out).exists():
                runs[-1]["results"] = json.loads(Path(out).read_text())
        payload["runs"] = runs
        summary_path = f"/results/baselines_{slug}_vllm.json"
        Path(summary_path).write_text(json.dumps(payload, indent=2))
        results_volume.commit()
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
    return payload


@app.function(
    image=feasibility_image,
    gpu="H100",
    timeout=120 * 60,
    secrets=secrets,
    volumes={"/results": results_volume},
)
def run_mirage_smoke_remote() -> dict:
    os.chdir("/bench")
    slug = _gpu_slug()
    cmd = [
        sys.executable,
        "engines/probes/mirage_probe.py",
        "--mirage-home",
        "/opt/mirage",
        "--smoke",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    out_path = f"/results/mirage_smoke_{slug}.json"
    body: dict = {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
    try:
        body["probe"] = json.loads(proc.stdout.splitlines()[-1])
    except Exception:
        pass
    Path(out_path).write_text(json.dumps(body, indent=2))
    results_volume.commit()
    return body


@app.local_entrypoint()
def bench(
    model: str = "Qwen/Qwen3.6-27B-FP8",
    max_model_len: int = 8192,
    n_requests: int = 10,
):
    result = run_vllm_bench_remote.remote(
        model=model,
        max_model_len=max_model_len,
        n_requests=n_requests,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def mirage_smoke():
    result = run_mirage_smoke_remote.remote()
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def feasibility(
    base_url: str | None = None,
    engine: str | None = None,
    smoke_mirage: bool = False,
):
    engines = [engine] if engine else None
    result = run_feasibility_remote.remote(base_url=base_url, engine=engines, smoke_mirage=smoke_mirage)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def sweep(
    base_url: str,
    engine: str = "vllm",
    model: str | None = None,
    workload: str | None = None,
):
    workloads = [workload] if workload else None
    result = run_sweep_remote.remote(base_url=base_url, engine=engine, model=model, workload=workloads)
    print(json.dumps(result, indent=2))
