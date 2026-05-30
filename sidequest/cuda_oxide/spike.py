"""Independent sidequest: can NVIDIA cuda-oxide (Rust->PTX, alpha, released 2026-05-09)
build + compile + run a GPU kernel on Modal?

Self-contained on purpose — NO imports from the (currently refactoring) harness. Own
Modal app + image. Scoped to the hackathon workspace. This only answers the hour-one
question: does the toolchain stand up at all? If yes, the Rust->PTX kernel-opt env is
live; if not, we fall back to Triton without burning the demo.

Run:
  cd ~/Desktop/autoresearch-hack
  export MODAL_PROFILE=... MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...   # from .env
  .venv/bin/modal run sidequest/cuda_oxide/spike.py
"""
import os

import modal
from dotenv import load_dotenv

load_dotenv("/Users/davidbai/Desktop/autoresearch-hack/.env")
assert os.environ.get("MODAL_PROFILE") == "autoresearch-hack", \
    f"refusing to run off the hackathon workspace (MODAL_PROFILE={os.environ.get('MODAL_PROFILE')!r})"

LLVM = "21"
NIGHTLY = "nightly-2026-04-03"
CARGO = "/root/.cargo/bin"

app = modal.App("ar2-cudaoxide-spike")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.2-devel-ubuntu24.04", add_python="3.11")
    .apt_install(
        "wget", "gnupg", "git", "curl", "build-essential", "ca-certificates",
        "software-properties-common", "lsb-release", "pkg-config", "libssl-dev",
    )
    # LLVM/Clang 21 from apt.llvm.org (Ubuntu 24.04 default tops out below 21).
    # noninteractive + NO "all" target — its extra pkgs trip systemd/resolvconf
    # postinst in Modal's minimal build container.
    .run_commands(
        "wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh && chmod +x /tmp/llvm.sh && DEBIAN_FRONTEND=noninteractive /tmp/llvm.sh 21",
        "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends clang-21 llvm-21 llvm-21-dev",
    )
    # pinned Rust nightly + components cuda-oxide requires
    .run_commands(
        f"curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain {NIGHTLY}",
        f"{CARGO}/rustup component add rust-src rustc-dev --toolchain {NIGHTLY}",
    )
    # the compiler subcommand + the repo (for its example kernels)
    .run_commands(
        f"{CARGO}/cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide",
        "git clone --depth 1 https://github.com/NVlabs/cuda-oxide.git /opt/cuda-oxide",
    )
    .env({
        "PATH": f"{CARGO}:/usr/lib/llvm-{LLVM}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "CUDA_OXIDE_LLC": f"/usr/bin/llc-{LLVM}",
    })
)


@app.function(image=image, gpu="H100", timeout=1200)
def spike() -> dict:
    import subprocess

    def run(cmd, cwd=None):
        r = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            shell=isinstance(cmd, str), timeout=600,
        )
        return {"rc": r.returncode, "out": (r.stdout + r.stderr)[-2500:]}

    res = {}
    res["gpu"] = run("nvidia-smi -L")
    res["llc"] = run("llc-21 --version | head -4")
    res["rustc"] = run("rustc --version")
    res["doctor"] = run(["cargo", "oxide", "doctor"], cwd="/opt/cuda-oxide")
    res["host_closure"] = run(["cargo", "oxide", "run", "host_closure"], cwd="/opt/cuda-oxide")
    res["gemm_sol"] = run(["cargo", "oxide", "run", "gemm_sol"], cwd="/opt/cuda-oxide")
    return res


@app.local_entrypoint()
def main():
    r = spike.remote()
    for k, v in r.items():
        print(f"\n===== {k}  (rc={v.get('rc')}) =====")
        print(v.get("out", "")[:1800])
