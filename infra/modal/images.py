"""AR² Modal image definitions — Agent #1 output.

Images are pure definitions; nothing is deployed or built here.
`MODAL_PROFILE` is read from the environment (via .env) so every Modal
call stays scoped to the hackathon workspace, never the work workspace.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

_HACKATHON_PROFILE = "autoresearch-hack"


def assert_hackathon_profile() -> None:
    """Raise if MODAL_PROFILE is not set to the hackathon workspace.

    Call before any billable Modal operation to ensure we never
    accidentally bill the work account.
    """
    profile = os.environ.get("MODAL_PROFILE", "")
    if profile != _HACKATHON_PROFILE:
        raise RuntimeError(
            f"MODAL_PROFILE must be '{_HACKATHON_PROFILE}', got '{profile}'. "
            "Set it in .env or the environment before running any Modal ops."
        )


import modal  # noqa: E402 — after load_dotenv so MODAL_PROFILE is already set

app = modal.App("ar2")

# ---------------------------------------------------------------------------
# base_image — shared foundation for all agents and sandboxes.
#
# Layers (baked in order, so each is cached independently):
#   1. debian_slim 3.12   — minimal base
#   2. apt_install        — system tools needed by coding agents + git ops
#   3. run_commands       — npm globals (codex + claude-code) + raindrop-ai
#   4. pip_install        — Python deps every AR² module uses
# ---------------------------------------------------------------------------
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "nodejs", "npm")
    .run_commands(
        "npm install -g @openai/codex @anthropic-ai/claude-code",
        "pip install raindrop-ai || true",  # best-effort; not on PyPI yet
    )
    .pip_install("pydantic", "httpx", "python-dotenv", "numpy", "modal")
)

# ---------------------------------------------------------------------------
# nanochat_gpu_image — extends base with karpathy/autoresearch v0.
#
# The `uv sync` + `uv run prepare.py` steps are baked so tokenizer / tiny
# dataset live in the image layer (no download at inference time).
#
# Example usage (do NOT call at import time):
#
#   @app.function(image=nanochat_gpu_image, gpu="H100", timeout=3600)
#   def train_nanochat(task: TaskSpec, budget: Budget) -> Submission:
#       ...
# ---------------------------------------------------------------------------
nanochat_gpu_image = (
    base_image
    .run_commands(
        "pip install uv",
        "git clone https://github.com/karpathy/autoresearch /opt/autoresearch",
        "cd /opt/autoresearch && uv sync",
        "cd /opt/autoresearch && uv run prepare.py",
    )
)

# matmul_gpu_image — lightweight CUDA image for kernel benchmarks (_modal_gpu_run).
matmul_gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "torch", "pydantic", "python-dotenv")
    .add_local_python_source("harness", "envs", "infra")
)

# vast_scorer_image — Modal CPU fn that SSH/SCP to a rented Vast GPU (_vast_gpu_run).
vast_scorer_image = (
    base_image
    .apt_install("openssh-client")
    .pip_install("vastai")
    .add_local_python_source("harness", "envs", "infra")
)

# ---------------------------------------------------------------------------
# Identical layer stack to base_image so the layer cache is shared;
# kept as a distinct name so harness code can reference it explicitly
# without importing nanochat_gpu_image's heavier layers.
# ---------------------------------------------------------------------------
sandbox_image = base_image.add_local_python_source("harness", "envs", "infra")
