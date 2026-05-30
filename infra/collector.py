"""AR² push telemetry collector — optional live-demo sink.

A single Modal FastAPI endpoint that sandboxes POST span batches to during
long runs. Spans are appended to a Modal Volume-backed JSONL file so the
outer dashboard can tail it in near-real-time.

This file is a DEFINITION ONLY. Nothing is deployed by importing it.
Call `assert_hackathon_profile()` before any modal deploy in a driver script.

Wire-format (one JSON object per line in the volume):
    {
        "trace_id": str,
        "version": int,
        "candidate": int | null,
        "env_id": str,
        "split": "train" | "heldout",
        "spans": [<raw span dicts from the in-sandbox shim>]
    }
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

import modal

from infra.modal.images import app, base_image

# ---------------------------------------------------------------------------
# Persistent storage — one volume, one JSONL file per run session.
# `create_if_missing=True` is safe to call at definition time; it is
# a no-op until the first actual Modal runtime interaction.
# ---------------------------------------------------------------------------
_spans_volume = modal.Volume.from_name("ar2-spans", create_if_missing=True)
_SPANS_PATH = "/data/spans.jsonl"


class SpanBatch(BaseModel):
    trace_id: str
    version: int
    candidate: int | None = None
    env_id: str
    split: str
    spans: list[dict[str, Any]]


@app.function(image=base_image, volumes={"/data": _spans_volume})
@modal.fastapi_endpoint(method="POST")
def ingest(batch: SpanBatch) -> dict[str, str]:
    """Receive a span batch from an in-sandbox telemetry shim and persist it.

    Sandboxes post here; the outer dashboard tails /data/spans.jsonl via
    the same Modal Volume mount.
    """
    import json

    line = batch.model_dump_json() + "\n"
    with open(_SPANS_PATH, "a") as f:
        f.write(line)
    _spans_volume.commit()
    return {"status": "ok", "trace_id": batch.trace_id}
