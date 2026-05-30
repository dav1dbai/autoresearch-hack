"""Fan out program.md candidates across Modal GPUs, one inner autoresearch run each.

The inner loop wants exactly one NVIDIA GPU and is fully self-contained -- that maps
one-to-one onto a Modal container, so N candidate strategies evaluate in parallel.
Reuses the determinate scoring in meta.py; only the orchestration differs.
"""

import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # local: MODAL_TOKEN_ID/SECRET so `modal run` auths to the right account

import modal

import meta

app = modal.App("meta-autoresearch")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "git clone --depth 1 https://github.com/karpathy/autoresearch /root/repo",
        # one-time data prep + tokenizer baked into the image so it isn't re-run per call
        "cd /root/repo && ~/.local/bin/uv sync && ~/.local/bin/uv run prepare.py",
    )
    .pip_install("pydantic")
    .add_local_python_source("meta")  # share row_count / score, no duplication
)

REPO = Path("/root/repo")


@app.function(image=image, gpu="H100", timeout=60 * 60)
def eval_program(program_md: str, tag: str, max_inner_iters: int = 6,
                 wall_clock_s: int = 1800) -> meta.InnerResult:
    import os
    import subprocess

    (REPO / "program.md").write_text(program_md)
    results = REPO / "results.tsv"
    proc = subprocess.Popen(
        [*os.environ.get("INNER_AGENT_CMD", "claude -p --dangerously-skip-permissions").split(),
         f"Read program.md and run the autoresearch loop on branch autoresearch/{tag}. "
         f"Do setup, then experiment. Do not ask me anything."],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    t0 = time.time()
    while proc.poll() is None:
        if meta.row_count(results) >= max_inner_iters or time.time() - t0 > wall_clock_s:
            break
        time.sleep(10)
    proc.terminate()
    return meta.score(results, time.time() - t0, str(REPO))


@app.local_entrypoint()
def main(candidates_dir: str = "candidates"):
    paths = sorted(Path(candidates_dir).glob("*.md"))
    programs = [p.read_text() for p in paths]
    tags = [p.stem for p in paths]
    for tag, res in zip(tags, eval_program.starmap(zip(programs, tags))):
        print(f"{res.best_val_bpb:.6f}  {tag}  ({res.n_experiments} exp, {res.crashed} crash)")
