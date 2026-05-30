from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .contracts import ExperimentRecord, RolloutResult, TaskSpec
from .domains import prepare_task
from .program import load_program


def _json_from_text(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
    for key in ("score", "correct", "reward"):
        for obj in reversed(candidates):
            if key in obj:
                return obj
    return candidates[-1] if candidates else {}


def _reward(score: dict[str, Any]) -> float:
    if "reward" in score:
        return float(score["reward"])
    if score.get("correct") is True and "speedup" in score:
        return float(score["speedup"])
    if "score" in score:
        return float(score["score"])
    return 0.0


def _read_experiments(path: Path) -> list[ExperimentRecord]:
    if not path.exists():
        return []
    records: list[ExperimentRecord] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(ExperimentRecord(**json.loads(line)))
        except Exception:
            continue
    return records


def _run_score(cmd: list[str], cwd: Path, timeout_s: int) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    score = _json_from_text(proc.stdout)
    if proc.returncode != 0:
        score.setdefault("error", proc.stderr[-2000:] or proc.stdout[-2000:])
        score.setdefault("returncode", proc.returncode)
    score.setdefault("stdout_tail", proc.stdout[-2000:])
    score.setdefault("stderr_tail", proc.stderr[-2000:])
    score.setdefault("reward", _reward(score))
    return score


def _ensure_agent_auth(cmd: str, cwd: Path) -> None:
    parts = shlex.split(cmd)
    if not parts or Path(parts[0]).name != "codex":
        return
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return
    subprocess.run(
        ["codex", "login", "--with-api-key"],
        input=api_key,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _copy_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def _mirror_workdir(workdir: Path, mirror_workdir: Path | None) -> None:
    if mirror_workdir is None:
        return
    for name in (
        "program.md",
        "experiments.jsonl",
        "agent.stdout.log",
        "agent.stderr.log",
        "task.py",
        "submission",
        "notes",
    ):
        _copy_path(workdir / name, mirror_workdir / name)


def run_inner_loop(
    spec: TaskSpec,
    run_root: str | Path,
    tag: str | None = None,
    agent_cmd: str | None = None,
    program_text: str | None = None,
    mirror_root: str | Path | None = None,
    sync_callback: Callable[[], None] | None = None,
    clean: bool = True,
) -> RolloutResult:
    tag = tag or spec.name
    run_root = Path(run_root)
    workdir = run_root / tag / "work"
    verifier_dir = run_root / tag / "verifier"
    mirror_tag = Path(mirror_root) / tag if mirror_root is not None else None
    mirror_workdir = mirror_tag / "work" if mirror_tag is not None else None
    if clean and (run_root / tag).exists():
        shutil.rmtree(run_root / tag)
    if clean and mirror_tag is not None and mirror_tag.exists():
        shutil.rmtree(mirror_tag)
    workdir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_task(spec, workdir, verifier_dir)
    outer_program = program_text if program_text is not None else load_program()
    (workdir / "program.md").write_text(
        f"{prepared.prompt}\n\nOuter-provided research methodology:\n\n{outer_program}\n"
    )
    (workdir / "experiments.jsonl").touch()
    _mirror_workdir(workdir, mirror_workdir)
    if sync_callback:
        sync_callback()

    cmd = agent_cmd or os.environ.get(
        "INNER_AGENT_CMD",
        "codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox",
    )
    prompt = (
        "Read ./program.md and run the iterative autoresearch loop for this task. "
        "Do not ask questions. Continue until you hit the written budget or the "
        "external runner stops you."
    )
    _ensure_agent_auth(cmd, workdir)

    t0 = time.time()
    crashed = False
    error: str | None = None
    proc: subprocess.Popen[str] | None = None
    stdout_log = workdir / "agent.stdout.log"
    stderr_log = workdir / "agent.stderr.log"
    agent_returncode: int | None = None
    try:
        with stdout_log.open("w") as out, stderr_log.open("w") as err:
            proc = subprocess.Popen(
                [*shlex.split(cmd), prompt],
                cwd=workdir,
                stdout=out,
                stderr=err,
                text=True,
            )
            while proc.poll() is None:
                experiments = _read_experiments(workdir / "experiments.jsonl")
                if len(experiments) >= spec.budget.max_iters:
                    break
                if time.time() - t0 > spec.budget.wall_clock_s:
                    break
                try:
                    proc.wait(timeout=spec.budget.poll_s)
                except subprocess.TimeoutExpired:
                    pass
                _mirror_workdir(workdir, mirror_workdir)
                if sync_callback:
                    sync_callback()
            agent_returncode = proc.poll()
    except Exception as exc:
        crashed = True
        error = str(exc)
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        if proc:
            agent_returncode = proc.poll()

    experiments = _read_experiments(workdir / "experiments.jsonl")
    _mirror_workdir(workdir, mirror_workdir)
    if sync_callback:
        sync_callback()
    final_score = _run_score(
        prepared.score_cmd,
        workdir,
        timeout_s=spec.budget.final_score_timeout_s,
    )
    final_reward = _reward(final_score)

    best_iter: int | None = None
    best_reward = final_reward
    for record in experiments:
        if record.reward >= best_reward:
            best_reward = record.reward
            best_iter = record.iter

    result = RolloutResult(
        task=spec,
        reward=best_reward,
        best_iter=best_iter,
        experiments=experiments,
        final_score=final_score,
        workdir=str(workdir),
        artifact_path=str(prepared.artifact_path),
        wall_clock_s=time.time() - t0,
        crashed=crashed or (agent_returncode not in (0, None)),
        error=error,
        agent_returncode=agent_returncode,
        agent_stdout_tail=stdout_log.read_text()[-4000:] if stdout_log.exists() else "",
        agent_stderr_tail=stderr_log.read_text()[-4000:] if stderr_log.exists() else "",
    )
    result_path = run_root / tag / "result.json"
    result_path.write_text(result.model_dump_json(indent=2))
    if mirror_tag is not None:
        _copy_path(result_path, mirror_tag / "result.json")
        _mirror_workdir(workdir, mirror_workdir)
    if sync_callback:
        sync_callback()
    return result
