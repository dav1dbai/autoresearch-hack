"""Basic double-loop executor on Modal sandboxes.

One rollout =
  1. spin up an *agent* sandbox (Claude Code or Codex headless) seeded with the
     task prompt, visible inputs, and an empty submission artifact;
  2. for each iteration: run one agent turn, read the submission out, then
     score it in a SEPARATE *verifier* sandbox, and write feedback back in;
  3. track the best reward.

The invariant: the verifier runs in its own sandbox that the agent's process
never touches, so the agent cannot read or edit the grader. The submission is
the only thing that crosses the boundary.

Run:
    uv run modal run harness/executor.py --task terminal-coding-rle
    uv run modal run harness/executor.py --task legal-loopholes-demo
    uv run modal run harness/executor.py --task kernelbench-square-matmul --agent-cmd "claude -p"
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import modal

from harness.schema import ImageSpec, TaskDef, Verifier, find_task

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_WORKDIR = "/work"
VERIFIER_DIR = "/verifier"

app = modal.App("autoresearch-harness")

# Agents talk to model providers; keys ride in via a folder-scoped .env secret.
SECRETS = [modal.Secret.from_dotenv()]


# --- images ---------------------------------------------------------------

def _default_agent_image() -> modal.Image:
    # Node 20 for the coding-agent CLIs; python for the submit tool + any local checks.
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "curl", "ca-certificates")
        .run_commands(
            "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
            "apt-get install -y nodejs",
            "npm install -g @anthropic-ai/claude-code @openai/codex || true",
        )
        .pip_install("pydantic>=2")
    )


def build_image(spec: ImageSpec | None) -> modal.Image:
    if spec is None:
        return modal.Image.debian_slim(python_version="3.12").pip_install("pydantic>=2")
    if spec.base == "debian_slim":
        img = modal.Image.debian_slim(python_version=spec.python_version)
    else:
        img = modal.Image.from_registry(spec.base, add_python=spec.python_version)
    if spec.apt:
        img = img.apt_install(*spec.apt)
    if spec.pip:
        img = img.pip_install(*spec.pip)
    if spec.run:
        img = img.run_commands(*spec.run)
    return img


# --- sandbox file helpers --------------------------------------------------

def _write_bytes(sb: modal.Sandbox, path: str, data: bytes) -> None:
    sb.exec("mkdir", "-p", str(Path(path).parent)).wait()
    with sb.open(path, "wb") as f:
        f.write(data)


def _write_text(sb: modal.Sandbox, path: str, text: str) -> None:
    _write_bytes(sb, path, text.encode())


def _read_bytes(sb: modal.Sandbox, path: str) -> bytes:
    with sb.open(path, "rb") as f:
        return f.read()


def _json_from_text(text: str) -> dict:
    """Grab the last top-level JSON object printed to stdout."""
    decoder = json.JSONDecoder()
    best: dict = {}
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            best = obj
    return best


# --- the submission tool (lives inside the agent sandbox) ------------------

SUBMIT_TOOL = """#!/usr/bin/env bash
# `submit` — the agent's handoff to the (out-of-reach) verifier.
# It validates that the submission artifact exists, then marks the turn done.
set -e
if [ ! -s "$SUBMISSION_PATH" ]; then
  echo "submit: nothing at $SUBMISSION_PATH — write your answer there first" >&2
  exit 1
fi
touch "$WORKDIR/.submitted"
echo "submit: staged $SUBMISSION_PATH for grading"
"""


# --- seeding & prompts -----------------------------------------------------

def _seed_agent_sandbox(sb: modal.Sandbox, task: TaskDef) -> None:
    submission_path = f"{AGENT_WORKDIR}/{task.submission.path}"
    if task.submission.template is not None:
        _write_text(sb, submission_path, task.submission.template)
    for fm in task.inputs:
        _write_bytes(sb, f"{AGENT_WORKDIR}/{fm.dest}", (REPO_ROOT / fm.src).read_bytes())

    _write_text(sb, "/usr/local/bin/submit", SUBMIT_TOOL)
    sb.exec("chmod", "+x", "/usr/local/bin/submit").wait()
    _write_text(sb, f"{AGENT_WORKDIR}/PROMPT.md", task.prompt)


def _turn_prompt(task: TaskDef, it: int, last: dict | None) -> str:
    parts = [
        f"Read ./PROMPT.md. This is iteration {it + 1} of at most {task.budget.max_iters}.",
        f"Write your answer to ./{task.submission.path}, then run `submit`.",
        "Do not ask questions; make your best edit and submit.",
    ]
    if last is not None:
        parts.append(
            "Feedback from the verifier on your previous submission "
            f"(reward {last.get('reward')}, higher is better):\n"
            f"{json.dumps(last, indent=2)[:2000]}"
        )
    return "\n\n".join(parts)


# --- verifier (separate sandbox) ------------------------------------------

def run_verifier(task: TaskDef, submission: bytes) -> dict:
    verifier: Verifier = task.verifier
    image = build_image(verifier.image)
    sb = modal.Sandbox.create(
        app=app,
        image=image,
        gpu=verifier.gpu,
        timeout=task.budget.verifier_timeout_s + 120,
        workdir=VERIFIER_DIR,
        block_network=True,  # deterministic graders need no network
    )
    try:
        for fm in verifier.files:
            _write_bytes(sb, f"{VERIFIER_DIR}/{fm.dest}", (REPO_ROOT / fm.src).read_bytes())
        sub_name = Path(task.submission.path).name
        _write_bytes(sb, f"{VERIFIER_DIR}/{sub_name}", submission)

        argv = [a.replace("{submission}", sub_name) for a in verifier.command]
        proc = sb.exec(*argv, workdir=VERIFIER_DIR, timeout=task.budget.verifier_timeout_s)
        proc.wait()
        out = proc.stdout.read()
        err = proc.stderr.read()
        score = _json_from_text(out)
        if not score:
            score = {"reward": 0.0, "error": (err or out)[-2000:]}
        score.setdefault("returncode", proc.returncode)
        return score
    finally:
        sb.terminate()


# --- the loop --------------------------------------------------------------

def run_rollout(task: TaskDef, agent_cmd: str | None = None) -> dict:
    agent_image = build_image(task.agent.image) if task.agent.image else _default_agent_image()
    cmd = agent_cmd or task.agent.cmd

    agent_sb = modal.Sandbox.create(
        app=app,
        image=agent_image,
        gpu=task.agent.gpu,
        timeout=task.budget.wall_clock_s + 300,
        workdir=AGENT_WORKDIR,
        secrets=SECRETS,
    )
    submission_path = f"{AGENT_WORKDIR}/{task.submission.path}"
    history: list[dict] = []
    best = {"reward": 0.0, "iter": -1}

    try:
        _seed_agent_sandbox(agent_sb, task)
        last: dict | None = None
        for it in range(task.budget.max_iters):
            prompt = _turn_prompt(task, it, last)
            agent_line = (
                f"cd {AGENT_WORKDIR} && export SUBMISSION_PATH={shlex.quote(submission_path)} "
                f"WORKDIR={AGENT_WORKDIR} && {cmd} {shlex.quote(prompt)}"
            )
            agent_sb.exec(
                "bash", "-lc", agent_line, timeout=task.budget.agent_turn_timeout_s
            ).wait()

            submission = _read_bytes(agent_sb, submission_path)
            score = run_verifier(task, submission)
            reward = float(score.get(task.verifier.reward_key, 0.0) or 0.0)
            last = {"reward": reward, **score}
            history.append({"iter": it, "reward": reward, "score": score})
            if reward > best["reward"]:
                best = {"reward": reward, "iter": it}

            _write_text(agent_sb, f"{AGENT_WORKDIR}/feedback.json", json.dumps(last, indent=2))
            print(f"[{task.id}] iter {it}: reward={reward}")
            if reward >= 1.0:
                break
    finally:
        agent_sb.terminate()

    result = {"task": task.id, "best": best, "history": history}
    return result


@app.local_entrypoint()
def main(task: str = "terminal-coding-rle", agent_cmd: str | None = None) -> None:
    td = find_task(task)
    result = run_rollout(td, agent_cmd)
    print(json.dumps(result, indent=2))
