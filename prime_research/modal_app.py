from __future__ import annotations

import os
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import modal

from prime_research.config import DEFAULT_CONFIG, load_tasks
from prime_research.program import DEFAULT_PROGRAM


app = modal.App("kernel-legal-autoresearch")
log_volume = modal.Volume.from_name("kernel-legal-autoresearch-logs", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("pydantic>=2", "python-dotenv>=1")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @openai/codex@0.135.0",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "~/.local/bin/uv tool install prime || true",
        "~/.local/bin/uv tool install verifiers || true",
    )
    .add_local_python_source("prime_research")
    .add_local_dir("inner-loop/domains", remote_path="/root/inner-loop/domains")
)

kernel_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install("pydantic>=2", "python-dotenv>=1", "numpy", "triton==3.2.0")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @openai/codex@0.135.0",
        "python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
        "git clone --depth 1 https://github.com/ScalingIntelligence/KernelBench /root/KernelBench",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "~/.local/bin/uv tool install prime || true",
        "~/.local/bin/uv tool install verifiers || true",
    )
    .add_local_python_source("prime_research")
    .add_local_dir("inner-loop/domains", remote_path="/root/inner-loop/domains")
)

def _codex_api_key() -> str | None:
    auth_path = Path.home() / ".codex" / "auth.json"
    if not auth_path.exists():
        return None
    try:
        value = json.loads(auth_path.read_text()).get("OPENAI_API_KEY")
    except Exception:
        return None
    return value or None


secret_values = {
    key: value
    for key in ("ANTHROPIC_API_KEY", "PRIME_API_KEY")
    if (value := os.environ.get(key))
}
if api_key := (_codex_api_key() or os.environ.get("OPENAI_API_KEY")):
    secret_values["OPENAI_API_KEY"] = api_key
secrets = [modal.Secret.from_dict(secret_values)] if secret_values else []


@app.function(
    image=base_image,
    timeout=30 * 60,
    secrets=secrets,
    volumes={"/mnt/autoresearch-logs": log_volume},
)
def run_legal_task(
    task_name: str,
    config_text: str,
    program_text: str,
    agent_cmd: str | None = None,
) -> dict:
    import os
    import tempfile

    from prime_research.config import select_tasks
    from prime_research.inner_loop import run_inner_loop

    os.chdir("/root")
    config_path = Path(tempfile.mkdtemp()) / "config.toml"
    config_path.write_text(config_text)
    task = select_tasks(config_path, [task_name])[0]
    result = run_inner_loop(
        task,
        "/root/runs",
        agent_cmd=agent_cmd,
        program_text=program_text,
        mirror_root="/mnt/autoresearch-logs/runs",
        sync_callback=log_volume.commit,
    )
    return result.model_dump()


@app.function(
    image=kernel_image,
    gpu="H100!",
    timeout=40 * 60,
    secrets=secrets,
    volumes={"/mnt/autoresearch-logs": log_volume},
)
def run_kernel_task(
    task_name: str,
    config_text: str,
    program_text: str,
    agent_cmd: str | None = None,
) -> dict:
    import os
    import tempfile

    from prime_research.config import select_tasks
    from prime_research.inner_loop import run_inner_loop

    os.environ.setdefault("KERNELBENCH_ROOT", "/root/KernelBench")
    os.chdir("/root")
    config_path = Path(tempfile.mkdtemp()) / "config.toml"
    config_path.write_text(config_text)
    task = select_tasks(config_path, [task_name])[0]
    result = run_inner_loop(
        task,
        "/root/runs",
        agent_cmd=agent_cmd,
        program_text=program_text,
        mirror_root="/mnt/autoresearch-logs/runs",
        sync_callback=log_volume.commit,
    )
    return result.model_dump()


@app.local_entrypoint()
def main(
    config: str = str(DEFAULT_CONFIG),
    task: str | None = None,
    program: str = str(DEFAULT_PROGRAM),
    agent_cmd: str | None = None,
):
    config_text = Path(config).read_text()
    program_text = Path(program).read_text()
    tasks = load_tasks(config)
    if task:
        tasks = [t for t in tasks if t.name == task]
    if not tasks:
        raise ValueError(f"No tasks selected from {config}")

    kernel_args = [
        (spec.name, config_text, program_text, agent_cmd)
        for spec in tasks
        if spec.domain == "kernel"
    ]
    legal_args = [
        (spec.name, config_text, program_text, agent_cmd)
        for spec in tasks
        if spec.domain == "legal"
    ]

    if kernel_args:
        for result in run_kernel_task.starmap(kernel_args):
            print(result)
    if legal_args:
        for result in run_legal_task.starmap(legal_args):
            print(result)
