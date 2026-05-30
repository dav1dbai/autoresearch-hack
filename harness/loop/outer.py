"""AR² harness — fixed outer loop (meta-optimizer driver).

Implements evaluate() and drive() from proof/DESIGN.md §3/§4.
All external dependencies (score_repo, load_ar) are injected so the module is
testable offline without Modal or real agents.

Injected-callable contracts
───────────────────────────
score_repo(ar_dir: Path, envs: list[Env], budget: Budget) -> list[Rollout]
    Run ar/entrypoint.solve on each env in a sandbox, return one Rollout per env.

load_ar(source_ref: str) -> types.ModuleType  (or any object with .improve)
    Load an ar/ snapshot by source_ref and return an object whose .improve method
    matches the Improve protocol: improve(archive, budget, spawn) -> Path.

Backend selection via AR2_BACKEND:
  local (default) — sequential evaluation loop; used for all offline tests.
  modal           — evaluates each generation's candidates in parallel via Modal,
                    using concurrent.futures.ThreadPoolExecutor to fan out evaluate()
                    calls that themselves dispatch via score_repo (which selects Modal
                    parallel rollouts when AR2_BACKEND=modal).
"""
from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
from typing import Callable

from harness.contracts import Archive, Attempt, Budget, Env, Rollout

_BACKEND = os.environ.get("AR2_BACKEND", "local")


# Lazy imports of real implementations — wrapped so the module loads offline.
def _default_score_repo(ar_dir: Path, envs: list[Env], budget: Budget, **kwargs) -> list[Rollout]:
    try:
        from harness.runtime.score import score_repo  # type: ignore[import]
        return score_repo(ar_dir, envs, budget, **kwargs)
    except ImportError as exc:
        raise RuntimeError("score_repo not available; inject it explicitly") from exc


def _default_load_ar(source_ref: str):
    try:
        from harness.runtime.loader import load_ar  # type: ignore[import]
        return load_ar(source_ref)
    except ImportError as exc:
        raise RuntimeError("load_ar not available; inject it explicitly") from exc


def evaluate(
    ar_dir: Path,
    train: list[Env],
    heldout: list[Env],
    budget: Budget,
    *,
    score_repo: Callable = _default_score_repo,
    version: int = 0,
    parent: int | None = None,
    diff_summary: str = "",
    source_ref: str = "",
    sync_traces: Path | None = Path("obs/traces.db"),
    archive_db: Path = Path("obs/archive.db"),
    mirror_workshop: bool | None = None,
) -> Attempt:
    """Evaluate one ar/ snapshot on train + heldout envs and return an Attempt.

    The meta-agent sees heldout reward ONLY as a scalar (DESIGN §11.3).
    Held-out env contents are never passed into improve().
    Persists inner curves on Attempt (D-03) and merges trace files when present.
    """
    from harness.loop.evaluate import attempt_from_rollouts
    from harness.runtime.loader import resolve_ar_dir
    from harness.util.progress import progress

    candidate = source_ref or str(ar_dir)
    try:
        ar_snapshot = resolve_ar_dir(candidate)
    except FileNotFoundError:
        ar_snapshot = Path(ar_dir)
    if score_repo is _default_score_repo:
        from harness.runtime.score import evaluate_rollouts

        progress(
            f"evaluate v{version}: scoring "
            f"({len(train)} train + {len(heldout)} heldout envs)"
        )
        train_rolls, heldout_rolls = evaluate_rollouts(
            ar_snapshot, train, heldout, budget,
            version=version, candidate=candidate,
        )
    else:
        progress(f"evaluate v{version}: train rollouts ({len(train)} envs)")
        train_rolls = score_repo(
            ar_snapshot, train, budget, version=version, candidate=candidate,
        )
        progress(f"evaluate v{version}: heldout rollouts ({len(heldout)} envs)")
        heldout_rolls = score_repo(
            ar_snapshot, heldout, budget, version=version, candidate=candidate,
        )

    attempt = attempt_from_rollouts(
        train_rolls,
        heldout_rolls,
        version=version,
        parent=parent,
        diff_summary=diff_summary,
        source_ref=candidate,
    )

    trace_paths = [
        Path(r.trace_path)
        for r in train_rolls + heldout_rolls
        if r.trace_path
    ]
    from harness.tracing import sync as db_sync

    if mirror_workshop is None:
        mirror_workshop = db_sync.workshop_enabled()

    if sync_traces is not None:
        sync_result = db_sync.sync_all(
            trace_files=trace_paths or None,
            attempt=attempt,
            live_push=mirror_workshop,
            traces_db=sync_traces,
            archive_db=archive_db,
        )
        if mirror_workshop:
            ws = db_sync.workshop_db_path()
            progress(
                f"evaluate v{version}: train={attempt.train_reward:.4f} "
                f"heldout={attempt.heldout_reward:.4f} → obs + {ws} "
                f"(+{sync_result['spans_inserted']} spans)"
            )
        else:
            progress(
                f"evaluate v{version}: train={attempt.train_reward:.4f} "
                f"heldout={attempt.heldout_reward:.4f} "
                f"→ {sync_traces} (+{sync_result['spans_inserted']} spans)"
            )
    else:
        progress(
            f"evaluate v{version}: train={attempt.train_reward:.4f} "
            f"heldout={attempt.heldout_reward:.4f}"
        )

    return attempt


def drive(
    ar0_dir: Path,
    train: list[Env],
    heldout: list[Env],
    budget: Budget,
    K: int,
    *,
    score_repo: Callable = _default_score_repo,
    load_ar: Callable = _default_load_ar,
    _persist_path: Path | None = None,
    _traces_db: Path | None = Path("obs/traces.db"),
    _archive_db: Path | None = Path("obs/archive.db"),
) -> Archive:
    """Fixed outer loop (DESIGN §4).  Evaluate v0, then for K generations:
    sample parents, call improve() once per parent, evaluate each candidate on
    train + heldout, archive, persist.

    Selection is on held-out reward.  Held-out env CONTENTS are never passed
    to improve() — only the scalar archive rewards (DESIGN §11.3).

    AR2_BACKEND=modal: each generation's candidates are evaluated in parallel
    via a ThreadPoolExecutor.  score_repo itself fans rollouts out across Modal
    containers when the backend is modal.
    AR2_BACKEND=local (default): sequential loop — offline-safe, test-friendly.
    """
    from harness.loop.archive import save  # local import to avoid circular at module level
    from harness.loop.snapshot import resolve_repo_root, smoke_check_version_snapshot
    from harness.util.progress import progress

    os.environ.setdefault("AR2_REPO_ROOT", str(resolve_repo_root()))

    def _noop_spawn(fn, list_of_args):
        return [fn(*((a,) if not isinstance(a, tuple) else a)) for a in list_of_args]

    progress(f"drive: backend={_BACKEND} K={K} budget={budget.wall_seconds}s")
    from harness.runtime.score import modal_backend_label

    mode = modal_backend_label()
    if mode is not None:
        progress(f"drive: modal app mode={mode}")

    _eval = lambda ar_dir, ver, par, diff, ref: evaluate(  # noqa: E731
        ar_dir, train, heldout, budget,
        score_repo=score_repo,
        version=ver,
        parent=par,
        diff_summary=diff,
        source_ref=ref,
        sync_traces=_traces_db,
        archive_db=_archive_db or Path("obs/archive.db"),
    )

    archive = Archive()
    progress("drive: evaluating v0 seed")
    try:
        from obs.events import log_event
        log_event("evaluate_start", "v0 seed evaluate", version=0)
    except ImportError:
        pass
    v0_attempt = _eval(ar0_dir, 0, None, "v0 seed", str(ar0_dir))
    archive.add(v0_attempt)
    try:
        from obs.events import log_event
        log_event(
            "evaluate_done",
            f"v0 train={v0_attempt.train_reward:.4f} heldout={v0_attempt.heldout_reward:.4f}",
            version=0,
            extra={"train": v0_attempt.train_reward, "heldout": v0_attempt.heldout_reward},
        )
    except ImportError:
        pass
    if _persist_path is not None:
        save(archive, _persist_path)
        progress(f"drive: archive persisted → {_persist_path}")

    version_counter = 1

    for gen in range(K):
        progress(f"drive: generation {gen + 1}/{K}")
        parent_k = int(os.environ.get("AR2_PARENT_K", "3"))
        parents = archive.sample_parents(parent_k, seed=gen)

        from harness.tracing.sync import prepare_improve_context

        improve_archive, workshop_traj = prepare_improve_context(archive)
        if workshop_traj:
            os.environ["AR2_WORKSHOP_TRAJECTORY"] = workshop_traj
            progress("drive: improve context loaded from Raindrop workshop")
        else:
            os.environ.pop("AR2_WORKSHOP_TRAJECTORY", None)

        # Collect all (cand_dir, parent_version) pairs — one improve() per parent.
        def _improve_one(parent_attempt: Attempt) -> list[tuple[Path, int]]:
            candidate_dirs: list[Path] = []
            workshop_traj = os.environ.get("AR2_WORKSHOP_TRAJECTORY", "")
            if _BACKEND == "modal":
                from harness.cloud.runner import run_improve_on_modal

                progress(f"drive: improve v{parent_attempt.version} on Modal")
                try:
                    from obs.events import log_event
                    log_event(
                        "improve_start",
                        f"meta-agent improve from v{parent_attempt.version}",
                        version=parent_attempt.version,
                        parent=parent_attempt.version,
                    )
                except ImportError:
                    pass
                cand_dir = run_improve_on_modal(
                    parent_attempt.source_ref,
                    improve_archive,
                    budget,
                    workshop_traj=workshop_traj,
                )
                ok, err = smoke_check_version_snapshot(cand_dir)
                if not ok:
                    progress(f"drive: skip broken Modal snapshot: {err}")
                    return []
                candidate_dirs.append(cand_dir)
                try:
                    from obs.events import log_event
                    log_event(
                        "improve_done",
                        f"candidate snapshot → {cand_dir.name}",
                        parent=parent_attempt.version,
                        path=cand_dir,
                    )
                except ImportError:
                    pass
            else:
                from harness.loop.snapshot import create_version_snapshot

                ar_obj = load_ar(parent_attempt.source_ref)
                version_root = create_version_snapshot(parent_attempt.source_ref)
                os.environ["AR2_VERSION_ROOT"] = str(version_root)
                try:
                    cand_dir = ar_obj.improve(improve_archive, budget, _noop_spawn)
                finally:
                    os.environ.pop("AR2_VERSION_ROOT", None)
                ok, err = smoke_check_version_snapshot(cand_dir)
                if not ok:
                    progress(f"drive: skip broken snapshot: {err}")
                    return []
                candidate_dirs.append(cand_dir)
            return [(d, parent_attempt.version) for d in candidate_dirs]

        all_candidates: list[tuple[Path, int]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(parents) or 1) as pool:
            futures = [pool.submit(_improve_one, p) for p in parents]
            for fut in concurrent.futures.as_completed(futures):
                all_candidates.extend(fut.result())

        # Evaluate candidates — parallel via ThreadPoolExecutor when backend=modal
        # (each evaluate() call dispatches its rollouts to Modal containers via
        # score_repo), sequential otherwise.
        def _eval_candidate(item: tuple[Path, int, int]) -> Attempt:
            cand_dir, parent_version, ver = item
            diff_summary = f"gen {gen} candidate from v{parent_version}"
            return _eval(cand_dir, ver, parent_version, diff_summary, str(cand_dir))

        # Assign version numbers before dispatch (deterministic ordering).
        versioned: list[tuple[Path, int, int]] = []
        for cand_dir, parent_version in all_candidates:
            versioned.append((cand_dir, parent_version, version_counter))
            version_counter += 1

        if _BACKEND == "modal" and len(versioned) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(versioned)) as pool:
                futures_eval = {pool.submit(_eval_candidate, item): item for item in versioned}
                # Collect in version order for deterministic archive ordering.
                results: dict[int, Attempt] = {}
                for fut in concurrent.futures.as_completed(futures_eval):
                    attempt = fut.result()
                    results[attempt.version] = attempt
            for _, _, ver in versioned:
                attempt = results[ver]
                archive.add(attempt)
                try:
                    from obs.events import log_event
                    log_event(
                        "evaluate_done",
                        f"v{attempt.version} train={attempt.train_reward:.4f} "
                        f"heldout={attempt.heldout_reward:.4f}",
                        version=attempt.version,
                        parent=attempt.parent,
                        extra={
                            "train": attempt.train_reward,
                            "heldout": attempt.heldout_reward,
                        },
                    )
                except ImportError:
                    pass
        else:
            for item in versioned:
                attempt = _eval_candidate(item)
                archive.add(attempt)
                try:
                    from obs.events import log_event
                    log_event(
                        "evaluate_done",
                        f"v{attempt.version} train={attempt.train_reward:.4f} "
                        f"heldout={attempt.heldout_reward:.4f}",
                        version=attempt.version,
                        parent=attempt.parent,
                        extra={
                            "train": attempt.train_reward,
                            "heldout": attempt.heldout_reward,
                        },
                    )
                except ImportError:
                    pass

        if _persist_path is not None:
            save(archive, _persist_path)
            progress(f"drive: generation {gen + 1} saved ({len(archive.attempts)} attempts)")

    progress(f"drive: complete — {len(archive.attempts)} attempts in archive")
    return archive
