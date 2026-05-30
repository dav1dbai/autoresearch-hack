from __future__ import annotations

import os
import shutil
from pathlib import Path

from .contracts import PreparedTask, TaskSpec
from .prompts import kernel_prompt, legal_prompt


ROOT = Path(__file__).resolve().parents[1]
LEGAL_DOMAIN = ROOT / "inner-loop" / "domains" / "legal" / "loopholes"
KERNEL_DOMAIN = ROOT / "inner-loop" / "domains" / "gpu" / "kernelbench"


def _chmod_readonly(path: Path) -> None:
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                child.chmod(0o444)
        path.chmod(0o555)
    else:
        path.chmod(0o444)


def _kernelbench_task_path(spec: TaskSpec) -> Path:
    if spec.task_path:
        return Path(spec.task_path).expanduser().resolve()

    root = Path(os.environ.get("KERNELBENCH_ROOT", "/root/KernelBench"))
    level = spec.task_level or "level1"
    task_dir = root / "KernelBench" / level
    tasks = sorted(task_dir.glob("*.py"))
    if not tasks:
        raise FileNotFoundError(f"No KernelBench tasks found in {task_dir}")
    idx = spec.task_index or 0
    if idx >= len(tasks):
        raise IndexError(f"task_index={idx} out of range for {task_dir} ({len(tasks)} tasks)")
    return tasks[idx]


def prepare_kernel_task(spec: TaskSpec, workdir: Path, verifier_dir: Path) -> PreparedTask:
    submission_dir = workdir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    (workdir / "notes").mkdir(exist_ok=True)

    task_src = _kernelbench_task_path(spec)
    visible_task = workdir / "task.py"
    verifier_task = verifier_dir / "task.py"
    shutil.copy2(task_src, visible_task)
    shutil.copy2(task_src, verifier_task)
    shutil.copy2(KERNEL_DOMAIN / "eval_submission.py", verifier_dir / "eval_submission.py")

    artifact = submission_dir / "submission.py"
    if not artifact.exists():
        artifact.write_text(
            "import torch\n\n"
            "class ModelNew(torch.nn.Module):\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        super().__init__()\n\n"
            "    def forward(self, *args, **kwargs):\n"
            "        raise NotImplementedError('Implement ModelNew for this task')\n"
        )

    _chmod_readonly(verifier_dir)
    score_cmd = [
        "python",
        str(verifier_dir / "eval_submission.py"),
        str(verifier_task),
        str(artifact),
    ]
    prepared = PreparedTask(
        spec=spec,
        workdir=workdir,
        verifier_dir=verifier_dir,
        artifact_path=artifact,
        score_cmd=score_cmd,
        prompt="",
    )
    prepared.prompt = kernel_prompt(prepared)
    return prepared


def prepare_legal_task(spec: TaskSpec, workdir: Path, verifier_dir: Path) -> PreparedTask:
    submission_dir = workdir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    (workdir / "notes").mkdir(exist_ok=True)

    task_src = Path(spec.task_path or LEGAL_DOMAIN / "tasks" / "demo_contract.md").resolve()
    visible_contract = workdir / "contract.md"
    verifier_contract = verifier_dir / "contract.md"
    shutil.copy2(task_src, visible_contract)
    shutil.copy2(task_src, verifier_contract)
    shutil.copy2(LEGAL_DOMAIN / "verify_findings.py", verifier_dir / "verify_findings.py")

    artifact = submission_dir / "findings.json"
    if not artifact.exists():
        artifact.write_text("[]\n")

    _chmod_readonly(verifier_dir)
    score_cmd = [
        "python",
        str(verifier_dir / "verify_findings.py"),
        "--contract",
        str(verifier_contract),
        "--findings",
        str(artifact),
    ]
    prepared = PreparedTask(
        spec=spec,
        workdir=workdir,
        verifier_dir=verifier_dir,
        artifact_path=artifact,
        score_cmd=score_cmd,
        prompt="",
    )
    prepared.prompt = legal_prompt(prepared)
    return prepared


def prepare_task(spec: TaskSpec, workdir: Path, verifier_dir: Path) -> PreparedTask:
    verifier_dir.mkdir(parents=True, exist_ok=True)
    if spec.domain == "kernel":
        return prepare_kernel_task(spec, workdir, verifier_dir)
    if spec.domain == "legal":
        return prepare_legal_task(spec, workdir, verifier_dir)
    raise ValueError(f"Unsupported domain: {spec.domain}")

