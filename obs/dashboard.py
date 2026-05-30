"""obs/dashboard.py — outer-loop dashboard for AR².

Renders obs/report.html from:
  - obs/archive.jsonl  (list of Attempt, one JSON object per line)
  - Raindrop Workshop DB (~/.raindrop/raindrop_workshop.db) or obs/traces.db fallback
    (SQLite: table `spans` with columns version, cost_usd, trace_id, ...)

Produces a self-contained HTML file with inline SVG:
  1. Two-colored outer curve: best held-out reward vs version (green=clean, red=hacked).
  2. Lineage tree: parent → child edges from Attempt.parent.
  3. Cost-per-version bar chart and train-vs-heldout gap per version.

Usage:
    from obs.dashboard import build_report
    build_report()                                   # default paths
    build_report(archive_path, db_path, out_path)   # custom paths
"""
from __future__ import annotations

import html
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from harness.contracts import Attempt, Archive, Budget

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DEFAULT_ARCHIVE = Path("obs/archive.jsonl")
_DEFAULT_DB = Path("obs/traces.db")
_DEFAULT_OUT = Path("obs/report.html")
_DEFAULT_EVENTS = Path("obs/run_events.jsonl")
_DEFAULT_VERSIONS = Path("versions")


def _resolve_cost_db(db_path: Path | None) -> Path:
    """Prefer Raindrop SOT when present; fall back to obs/traces.db."""
    if db_path is not None:
        return db_path
    from harness.tracing.sync import raindrop_db_path

    rd = raindrop_db_path()
    if rd.exists():
        return rd
    return _DEFAULT_DB


def _load_attempts(archive_path: Path) -> list[Attempt]:
    if not archive_path.exists():
        return []
    attempts = []
    with archive_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                attempts.append(Attempt.model_validate_json(line))
    return sorted(attempts, key=lambda a: a.version)


def _cost_per_version(db_path: Path) -> dict[int, float]:
    """Sum cost_usd from spans table grouped by version. Returns {} if db absent."""
    if not db_path.exists():
        return {}
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT version, SUM(cost_usd) FROM spans GROUP BY version"
        ).fetchall()
        return {int(r[0]): float(r[1] or 0.0) for r in rows}
    except sqlite3.OperationalError:
        return {}
    finally:
        con.close()


def _load_events(events_path: Path) -> list[dict]:
    if not events_path.exists():
        return []
    rows: list[dict] = []
    with events_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_tail(path: Path, max_lines: int = 60, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    tail = lines[-max_lines:]
    text = "\n".join(tail)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError):
        return str(ts)


def _run_timeline_html(events: list[dict]) -> str:
    if not events:
        return (
            "<p>No run events yet. Events appear in "
            "<code>obs/run_events.jsonl</code> during <code>drive()</code>.</p>"
        )
    rows = [
        "<table class='tbl'><thead><tr>"
        "<th>Time</th><th>Phase</th><th>Ver</th><th>Message</th>"
        "</tr></thead><tbody>"
    ]
    for ev in events[-80:]:
        phase = html.escape(str(ev.get("phase", "")))
        msg = html.escape(str(ev.get("message", "")))
        ver = ev.get("version")
        ver_s = f"v{ver}" if ver is not None else "—"
        rows.append(
            f"<tr><td>{_fmt_ts(ev.get('ts'))}</td>"
            f"<td><code>{phase}</code></td>"
            f"<td>{html.escape(ver_s)}</td>"
            f"<td>{msg}</td></tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _attempt_table_html(attempts: list[Attempt]) -> str:
    if not attempts:
        return "<p>No attempts yet.</p>"
    rows = [
        "<table class='tbl'><thead><tr>"
        "<th>Ver</th><th>Parent</th><th>Train</th><th>Held-out</th>"
        "<th>Gap</th><th>Hack?</th><th>Summary</th><th>Snapshot</th>"
        "</tr></thead><tbody>"
    ]
    for a in attempts:
        gap = a.train_reward - a.heldout_reward
        hacked = bool(a.hack_flags)
        hack_cell = (
            f"<span class='bad'>{html.escape('; '.join(a.hack_flags))}</span>"
            if hacked
            else "<span class='ok'>clean</span>"
        )
        snap = html.escape(Path(a.source_ref).name if a.source_ref else "—")
        rows.append(
            f"<tr id='attempt-v{a.version}'>"
            f"<td><strong>v{a.version}</strong></td>"
            f"<td>{'v' + str(a.parent) if a.parent is not None else '—'}</td>"
            f"<td>{a.train_reward:.4f}</td>"
            f"<td>{a.heldout_reward:.4f}</td>"
            f"<td>{gap:+.4f}</td>"
            f"<td>{hack_cell}</td>"
            f"<td>{html.escape(a.diff_summary)}</td>"
            f"<td><code>{snap}</code></td>"
            f"</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _inner_curves_html(attempts: list[Attempt]) -> str:
    if not attempts:
        return "<p>No inner curves yet.</p>"
    parts = ["<div class='inner-grid'>"]
    for a in attempts:
        rollouts = a.train_rollouts or []
        if not rollouts:
            parts.append(
                f"<div class='inner-card'><strong>v{a.version}</strong> "
                f"<span class='muted'>no train_rollouts recorded</span></div>"
            )
            continue
        for r in rollouts:
            rewards = r.rewards or [r.final_reward]
            curve = " → ".join(f"{x:.3f}" for x in rewards)
            parts.append(
                f"<div class='inner-card'>"
                f"<strong>v{a.version}</strong> "
                f"<code>{html.escape(r.env_id)}</code> "
                f"<span class='muted'>({len(rewards)} steps)</span><br/>"
                f"<span class='mono'>{html.escape(curve)}</span>"
                f"</div>"
            )
    parts.append("</div>")
    return "\n".join(parts)


def _improve_activity_html(
    attempts: list[Attempt],
    versions_root: Path,
) -> str:
    """Show improve_prompt excerpt + agent.log tail per non-v0 snapshot."""
    candidates = [a for a in attempts if a.parent is not None and a.source_ref]
    if not candidates:
        return (
            "<p>No improve candidates yet. After gen 1, each snapshot under "
            f"<code>{html.escape(str(versions_root))}/</code> gets "
            "<code>improve_prompt.md</code> + <code>agent.log</code>.</p>"
        )
    parts: list[str] = []
    for a in sorted(candidates, key=lambda x: x.version):
        root = Path(a.source_ref)
        if not root.is_absolute() and not root.exists():
            alt = versions_root / root.name
            root = alt if alt.exists() else root
        prompt_path = root / "improve_prompt.md"
        log_path = root / "agent.log"
        parts.append(f"<details class='improve-block' id='improve-v{a.version}'>")
        parts.append(
            f"<summary><strong>v{a.version}</strong> from v{a.parent} — "
            f"<code>{html.escape(root.name)}</code></summary>"
        )
        if prompt_path.exists():
            prompt = _read_tail(prompt_path, max_lines=40, max_chars=4000)
            parts.append("<h3>Improve prompt (tail)</h3>")
            parts.append(
                f"<pre class='log'>{html.escape(prompt or '(empty)')}</pre>"
            )
        else:
            parts.append(
                "<p class='muted'>No improve_prompt.md (run predates prompt capture).</p>"
            )
        if log_path.exists():
            log = _read_tail(log_path, max_lines=80, max_chars=12000)
            parts.append("<h3>Meta-agent log (tail)</h3>")
            parts.append(f"<pre class='log'>{html.escape(log or '(empty)')}</pre>")
        else:
            parts.append("<p class='muted'>No agent.log yet — improve may still be running.</p>")
        parts.append("</details>")
    return "\n".join(parts)


def _live_status_html(events: list[dict], attempts: list[Attempt]) -> str:
    last = events[-1] if events else None
    n_ver = len({a.version for a in attempts})
    if last:
        phase = html.escape(str(last.get("phase", "unknown")))
        msg = html.escape(str(last.get("message", "")))
        when = _fmt_ts(last.get("ts"))
        body = f"Last event <strong>{when}</strong>: <code>{phase}</code> — {msg}"
    elif attempts:
        body = f"Run idle — {len(attempts)} attempt(s) across {n_ver} version(s) in archive."
    else:
        body = "Waiting for first evaluate/improve event."
    return f"<div class='live-banner'>{body}</div>"


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

_PAD = 60  # px padding around each chart
_W = 600   # chart width (inner)
_H = 280   # chart height (inner)


def _svg_open(w: int, h: int, extra_attrs: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" {extra_attrs}>'
    )


def _axes(pad: int, w: int, h: int, x_labels: list[str], y_ticks: list[float]) -> str:
    parts: list[str] = []
    # axis lines
    parts.append(
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + h}" '
        f'stroke="#888" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{pad}" y1="{pad + h}" x2="{pad + w}" y2="{pad + h}" '
        f'stroke="#888" stroke-width="1"/>'
    )
    # x labels
    n = len(x_labels)
    if n > 1:
        for i, lbl in enumerate(x_labels):
            x = pad + int(i * w / (n - 1))
            y = pad + h + 18
            parts.append(
                f'<text x="{x}" y="{y}" text-anchor="middle" '
                f'font-size="11" fill="#555">{html.escape(str(lbl))}</text>'
            )
    elif n == 1:
        x = pad + w // 2
        y = pad + h + 18
        parts.append(
            f'<text x="{x}" y="{y}" text-anchor="middle" '
            f'font-size="11" fill="#555">{html.escape(str(x_labels[0]))}</text>'
        )
    # y ticks
    for v in y_ticks:
        y = pad + h - int(v * h)
        parts.append(
            f'<line x1="{pad - 4}" y1="{y}" x2="{pad}" y2="{y}" '
            f'stroke="#888" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad - 7}" y="{y + 4}" text-anchor="end" '
            f'font-size="10" fill="#555">{v:.1f}</text>'
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chart 1: two-colored outer curve
# ---------------------------------------------------------------------------

class _CurvePoint(NamedTuple):
    version: int
    reward: float
    hacked: bool
    trace_id: str
    hack_flags: list[str]


def _outer_curve_svg(attempts: list[Attempt]) -> str:
    if not attempts:
        return "<p>No attempts yet.</p>"

    # best held-out per version (prefer clean)
    by_version: dict[int, Attempt] = {}
    for a in attempts:
        v = a.version
        if v not in by_version:
            by_version[v] = a
        else:
            prev = by_version[v]
            # prefer clean over hacked; then higher reward
            if (not a.hack_flags and prev.hack_flags) or (
                bool(a.hack_flags) == bool(prev.hack_flags)
                and a.heldout_reward > prev.heldout_reward
            ):
                by_version[v] = a

    versions = sorted(by_version)
    points = [
        _CurvePoint(
            version=v,
            reward=by_version[v].heldout_reward,
            hacked=bool(by_version[v].hack_flags),
            trace_id=by_version[v].trace_id,
            hack_flags=by_version[v].hack_flags,
        )
        for v in versions
    ]

    pad, w, h = _PAD, _W, _H
    total_w, total_h = w + 2 * pad, h + 2 * pad

    def _px(i: int, r: float) -> tuple[int, int]:
        n = len(points)
        x = pad + int(i * w / max(n - 1, 1))
        y = pad + h - int(r * h)
        return x, y

    parts: list[str] = [_svg_open(total_w, total_h)]
    parts.append(
        _axes(
            pad, w, h,
            x_labels=[f"v{p.version}" for p in points],
            y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
    )

    # axis labels
    parts.append(
        f'<text x="{pad + w // 2}" y="{total_h - 4}" text-anchor="middle" '
        f'font-size="12" fill="#333">Version</text>'
    )
    parts.append(
        f'<text x="14" y="{pad + h // 2}" text-anchor="middle" '
        f'font-size="12" fill="#333" '
        f'transform="rotate(-90,14,{pad + h // 2})">Held-out reward</text>'
    )

    # connect consecutive same-color segments
    i = 0
    while i < len(points) - 1:
        x0, y0 = _px(i, points[i].reward)
        x1, y1 = _px(i + 1, points[i + 1].reward)
        color = "#c0392b" if points[i].hacked or points[i + 1].hacked else "#27ae60"
        parts.append(
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
            f'stroke="{color}" stroke-width="2.5" stroke-opacity="0.7"/>'
        )
        i += 1

    # dots with tooltip title
    for i, p in enumerate(points):
        x, y = _px(i, p.reward)
        color = "#c0392b" if p.hacked else "#27ae60"
        flags_text = "; ".join(p.hack_flags) if p.hack_flags else "clean"
        tip = f"v{p.version}: reward={p.reward:.3f} | {flags_text}"
        # drilldown link if trace_id present
        if p.trace_id:
            parts.append(f'<a href="#trace-{html.escape(p.trace_id)}">')
        parts.append(f"<title>{html.escape(tip)}</title>")
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="6" fill="{color}" '
            f'stroke="white" stroke-width="1.5">'
            f"<title>{html.escape(tip)}</title></circle>"
        )
        if p.trace_id:
            parts.append("</a>")
        # label above
        label_color = color
        parts.append(
            f'<text x="{x}" y="{y - 10}" text-anchor="middle" '
            f'font-size="10" fill="{label_color}">{p.reward:.2f}</text>'
        )

    # legend
    lx = pad + w - 120
    ly = pad + 10
    parts.append(
        f'<circle cx="{lx}" cy="{ly}" r="5" fill="#27ae60"/>'
        f'<text x="{lx + 9}" y="{ly + 4}" font-size="11" fill="#333">clean gain</text>'
    )
    parts.append(
        f'<circle cx="{lx}" cy="{ly + 16}" r="5" fill="#c0392b"/>'
        f'<text x="{lx + 9}" y="{ly + 20}" font-size="11" fill="#333">hacked</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chart 2: lineage tree (parent → child)
# ---------------------------------------------------------------------------

def _lineage_svg(attempts: list[Attempt]) -> str:
    if not attempts:
        return "<p>No attempts yet.</p>"

    # build children map
    children: dict[int | None, list[Attempt]] = {}
    for a in attempts:
        children.setdefault(a.parent, []).append(a)

    # BFS to assign positions
    # nodes keyed by version; x = depth column, y = sibling row
    NodePos = dict[int, tuple[int, int]]  # version -> (col, row)
    pos: NodePos = {}
    row_counter = [0]

    def _layout(version: int | None, col: int) -> None:
        kids = sorted(children.get(version, []), key=lambda a: a.version)
        for kid in kids:
            pos[kid.version] = (col, row_counter[0])
            row_counter[0] += 1
            _layout(kid.version, col + 1)

    _layout(None, 0)

    if not pos:
        return "<p>No lineage.</p>"

    max_col = max(c for c, _ in pos.values())
    max_row = max(r for _, r in pos.values())

    cell_w, cell_h = 130, 56
    pad = 20
    total_w = max(400, pad * 2 + (max_col + 1) * cell_w)
    total_h = max(120, pad * 2 + (max_row + 1) * cell_h)

    attempt_map = {a.version: a for a in attempts}

    def _cx(col: int) -> int:
        return pad + col * cell_w + cell_w // 2

    def _cy(row: int) -> int:
        return pad + row * cell_h + cell_h // 2

    parts: list[str] = [_svg_open(total_w, total_h)]

    # edges
    for a in attempts:
        if a.parent is not None and a.parent in pos and a.version in pos:
            pc, pr = pos[a.parent]
            cc, cr = pos[a.version]
            x0, y0 = _cx(pc), _cy(pr)
            x1, y1 = _cx(cc), _cy(cr)
            parts.append(
                f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
                f'stroke="#aaa" stroke-width="1.5" stroke-dasharray="4,2"/>'
            )

    # nodes
    node_w, node_h = 100, 44
    for v, (col, row) in pos.items():
        a = attempt_map[v]
        cx, cy = _cx(col), _cy(row)
        x0, y0 = cx - node_w // 2, cy - node_h // 2
        hacked = bool(a.hack_flags)
        fill = "#fde8e8" if hacked else "#e8f5e9"
        border = "#c0392b" if hacked else "#27ae60"
        parts.append(
            f'<rect x="{x0}" y="{y0}" width="{node_w}" height="{node_h}" '
            f'rx="6" fill="{fill}" stroke="{border}" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{cx}" y="{cy - 8}" text-anchor="middle" '
            f'font-size="11" font-weight="bold" fill="#222">v{v}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{cy + 6}" text-anchor="middle" '
            f'font-size="10" fill="#444">R={a.heldout_reward:.3f}</text>'
        )
        if hacked:
            tip = "; ".join(a.hack_flags)
            parts.append(
                f'<text x="{cx}" y="{cy + 19}" text-anchor="middle" '
                f'font-size="9" fill="#c0392b">⚑ hacked</text>'
            )
            parts.append(f"<title>{html.escape(tip)}</title>")
        else:
            parts.append(
                f'<text x="{cx}" y="{cy + 19}" text-anchor="middle" '
                f'font-size="9" fill="#27ae60">✓ clean</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chart 3: cost-per-version bars + train-vs-heldout gap
# ---------------------------------------------------------------------------

def _cost_gap_svg(attempts: list[Attempt], cost_per_version: dict[int, float]) -> str:
    versions = sorted({a.version for a in attempts})
    if not versions:
        return "<p>No data.</p>"

    attempt_map: dict[int, list[Attempt]] = {}
    for a in attempts:
        attempt_map.setdefault(a.version, []).append(a)

    def _best(vs: list[Attempt]) -> Attempt:
        clean = [a for a in vs if not a.hack_flags]
        pool = clean or vs
        return max(pool, key=lambda a: a.heldout_reward)

    best_per = {v: _best(attempt_map[v]) for v in versions}
    costs = [cost_per_version.get(v, best_per[v].cost.usd or 0.0) for v in versions]
    gaps = [best_per[v].train_reward - best_per[v].heldout_reward for v in versions]

    n = len(versions)
    pad, w, h = _PAD, _W, _H
    total_w, total_h = w + 2 * pad, h + 2 * pad

    max_cost = max(costs) if any(c > 0 for c in costs) else 1.0
    bar_w = max(8, w // (n * 2 + 1))

    parts: list[str] = [_svg_open(total_w, total_h)]
    parts.append(
        _axes(
            pad, w, h,
            x_labels=[f"v{v}" for v in versions],
            y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
    )

    spacing = w // max(n, 1)

    for i, v in enumerate(versions):
        cx = pad + i * spacing + spacing // 2
        # cost bar (blue)
        cost_norm = costs[i] / max_cost if max_cost > 0 else 0
        bar_h = int(cost_norm * h * 0.8)
        bx = cx - bar_w - 2
        by = pad + h - bar_h
        parts.append(
            f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bar_h}" '
            f'fill="#3498db" fill-opacity="0.7">'
            f'<title>v{v} cost: ${costs[i]:.4f}</title></rect>'
        )
        parts.append(
            f'<text x="{bx + bar_w // 2}" y="{by - 3}" text-anchor="middle" '
            f'font-size="9" fill="#2980b9">${costs[i]:.3f}</text>'
        )
        # gap bar (orange for positive gap = train > heldout = possible overfit)
        gap = gaps[i]
        gap_norm = min(abs(gap), 1.0)
        gap_bh = int(gap_norm * h * 0.8)
        gx = cx + 2
        gap_fill = "#e67e22" if gap > 0.05 else "#95a5a6"
        gy = pad + h - gap_bh
        parts.append(
            f'<rect x="{gx}" y="{gy}" width="{bar_w}" height="{gap_bh}" '
            f'fill="{gap_fill}" fill-opacity="0.7">'
            f'<title>v{v} train-heldout gap: {gap:+.3f}</title></rect>'
        )
        parts.append(
            f'<text x="{gx + bar_w // 2}" y="{gy - 3}" text-anchor="middle" '
            f'font-size="9" fill="{gap_fill}">{gap:+.2f}</text>'
        )

    # legend
    lx = pad + w - 160
    ly = pad + 10
    parts.append(
        f'<rect x="{lx}" y="{ly - 7}" width="12" height="12" fill="#3498db" fill-opacity="0.7"/>'
        f'<text x="{lx + 16}" y="{ly + 4}" font-size="11" fill="#333">cost (USD, scaled)</text>'
    )
    parts.append(
        f'<rect x="{lx}" y="{ly + 9}" width="12" height="12" fill="#e67e22" fill-opacity="0.7"/>'
        f'<text x="{lx + 16}" y="{ly + 20}" font-size="11" fill="#333">train−heldout gap</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Trace drilldown anchor blocks
# ---------------------------------------------------------------------------

def _trace_anchors(attempts: list[Attempt]) -> str:
    hacked = [a for a in attempts if a.hack_flags and a.trace_id]
    if not hacked:
        return ""
    rows = ["<h2>Hacked-version trace drilldowns</h2><ul>"]
    for a in hacked:
        tid = html.escape(a.trace_id)
        flags = html.escape("; ".join(a.hack_flags))
        rows.append(
            f'<li id="trace-{tid}">'
            f"<strong>v{a.version}</strong> — trace <code>{tid}</code> — "
            f"flags: <span style='color:#c0392b'>{flags}</span>"
            f"</li>"
        )
    rows.append("</ul>")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

def build_report(
    archive_path: Path = _DEFAULT_ARCHIVE,
    db_path: Path | None = None,
    out: Path = _DEFAULT_OUT,
    events_path: Path = _DEFAULT_EVENTS,
    versions_root: Path = _DEFAULT_VERSIONS,
    auto_refresh_seconds: int | None = None,
) -> Path:
    """Render obs/report.html from archive.jsonl + span cost DB + run events."""
    attempts = _load_attempts(archive_path)
    cost_per_version = _cost_per_version(_resolve_cost_db(db_path))
    events = _load_events(events_path)

    outer_curve = _outer_curve_svg(attempts)
    lineage = _lineage_svg(attempts)
    cost_gap = _cost_gap_svg(attempts, cost_per_version)
    drilldowns = _trace_anchors(attempts)
    live_status = _live_status_html(events, attempts)
    timeline = _run_timeline_html(events)
    attempt_table = _attempt_table_html(attempts)
    inner_curves = _inner_curves_html(attempts)
    improve_activity = _improve_activity_html(attempts, versions_root)

    n_attempts = len(attempts)
    n_versions = len({a.version for a in attempts})
    n_hacked = sum(1 for a in attempts if a.hack_flags)

    from obs.metrics import S, delta_S

    delta_rows = []
    for a in attempts:
        ds = delta_S(Archive(attempts=attempts), a.version)
        if ds is not None:
            delta_rows.append(
                f"<tr><td>v{a.version}</td><td>{S(a):+.4f}</td>"
                f"<td>{ds:+.4f}</td></tr>"
            )
    delta_table = (
        "<table class='tbl'><thead><tr><th>Ver</th><th>S(N)</th><th>ΔS(N)</th></tr></thead>"
        f"<tbody>{''.join(delta_rows) or '<tr><td colspan=3>No ΔS yet</td></tr>'}</tbody></table>"
    )

    built_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    refresh_meta = ""
    if auto_refresh_seconds and auto_refresh_seconds > 0:
        refresh_meta = (
            f'<meta http-equiv="refresh" content="{int(auto_refresh_seconds)}"/>'
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
{refresh_meta}
<title>AR² outer-loop dashboard</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 32px auto; color: #222; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
  .meta {{ color: #666; font-size: 0.85rem; margin-bottom: 28px; }}
  h2 {{ font-size: 1.1rem; margin-top: 36px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  h3 {{ font-size: 0.95rem; margin: 12px 0 6px; }}
  svg {{ display: block; margin: 12px 0; overflow: visible; }}
  .live-banner {{ background: #eef6ff; border: 1px solid #b8daff; padding: 10px 14px;
    border-radius: 6px; margin-bottom: 20px; font-size: 0.9rem; }}
  .tbl {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .tbl th, .tbl td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
  .tbl th {{ background: #f5f5f5; }}
  .ok {{ color: #27ae60; }}
  .bad {{ color: #c0392b; font-size: 0.8rem; }}
  .muted {{ color: #888; font-size: 0.85rem; }}
  .mono {{ font-family: ui-monospace, monospace; font-size: 0.8rem; }}
  .log {{ background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px;
    overflow-x: auto; font-size: 0.75rem; line-height: 1.4; white-space: pre-wrap; }}
  .inner-grid {{ display: grid; gap: 10px; }}
  .inner-card {{ border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px 12px;
    font-size: 0.85rem; }}
  details.improve-block {{ border: 1px solid #ddd; border-radius: 6px; padding: 8px 12px;
    margin-bottom: 10px; }}
  details.improve-block summary {{ cursor: pointer; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>AR² outer-loop dashboard</h1>
<p class="meta">
  Built {built_at} &mdash;
  {n_attempts} attempt(s) across {n_versions} version(s) &mdash;
  {n_hacked} hacked &mdash; {n_attempts - n_hacked} clean
  {" &mdash; auto-refresh " + str(auto_refresh_seconds) + "s" if auto_refresh_seconds else ""}
</p>

{live_status}

<h2>Live run timeline</h2>
<p style="font-size:0.85rem;color:#555">
  Append-only log from <code>obs/run_events.jsonl</code>. Shows evaluate/improve milestones
  as the outer loop runs. Re-run <code>uv run python -m obs.dashboard</code> or open with
  auto-refresh during long Modal runs.
</p>
{timeline}

<h2>Attempt detail</h2>
{attempt_table}

<h2>What the improve agent did</h2>
<p style="font-size:0.85rem;color:#555">
  Each generation: copy parent snapshot → build prompt from archive + workshop traces →
  run <code>MUTATE_AGENT_CMD</code> (Codex) → evaluate new <code>ar/</code> on Modal.
  Expand a row for the prompt and meta-agent transcript.
</p>
{improve_activity}

<h2>Inner solve curves (train rollouts)</h2>
<p style="font-size:0.85rem;color:#555">
  Per-env reward sequence inside each version's <code>solve()</code> loop (baseline → edits).
</p>
{inner_curves}

<h2>ΔS(N) — inner slope gain (D-04)</h2>
{delta_table}

<h2>Outer curve: held-out reward vs version</h2>
<p style="font-size:0.85rem;color:#555">
  Green = genuine gain (no hack flags). Red = hacked (hack_flags non-empty).
  Click a red dot to jump to its trace drilldown.
</p>
{outer_curve}

<h2>Lineage tree (version archive)</h2>
{lineage}

<h2>Cost per version &amp; train–held-out gap</h2>
<p style="font-size:0.85rem;color:#555">
  Blue bars = total cost_usd (from Raindrop spans or attempt.cost.usd, scaled to max).
  Orange bars = train_reward − heldout_reward gap (&gt;0.05 flags possible overfit).
</p>
{cost_gap}

{drilldowns}
</body>
</html>
"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    return out


if __name__ == "__main__":
    import os

    refresh = int(os.environ.get("AR2_DASHBOARD_REFRESH", "0") or "0") or None
    built = build_report(
        archive_path=Path(os.environ.get("AR2_DASHBOARD_ARCHIVE", str(_DEFAULT_ARCHIVE))),
        db_path=Path(os.environ.get("AR2_DASHBOARD_DB", str(_DEFAULT_DB))),
        out=Path(os.environ.get("AR2_DASHBOARD_OUT", str(_DEFAULT_OUT))),
        events_path=Path(os.environ.get("AR2_DASHBOARD_EVENTS", str(_DEFAULT_EVENTS))),
        versions_root=Path(os.environ.get("AR2_DASHBOARD_VERSIONS", str(_DEFAULT_VERSIONS))),
        auto_refresh_seconds=refresh,
    )
    print(f"Report written to {built}")
