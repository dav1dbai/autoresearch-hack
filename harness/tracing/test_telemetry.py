"""Offline tests for harness/tracing/telemetry.py."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from harness.tracing.telemetry import inject, parse_trace_file, write_span


@pytest.fixture()
def tmp(tmp_path: Path) -> Path:
    return tmp_path


class TestInject:
    def test_returns_required_keys(self):
        tags = {"version": 1, "candidate": "c1", "env_id": "e1", "split": "train", "trace_id": "abc"}
        env = inject(tags)
        assert env["AR2_TRACE_ID"] == "abc"
        assert env["AR2_VERSION"] == "1"
        assert env["AR2_CANDIDATE"] == "c1"
        assert env["AR2_ENV_ID"] == "e1"
        assert env["AR2_SPLIT"] == "train"

    def test_generates_trace_id_when_missing(self):
        env = inject({"version": 0, "candidate": "x", "env_id": "e", "split": "train"})
        assert env["AR2_TRACE_ID"]

    def test_has_base_url_keys(self):
        env = inject({})
        assert "OPENAI_BASE_URL" in env
        assert "ANTHROPIC_BASE_URL" in env
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" in env

    def test_forward_gpu_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AR2_GPU_BACKEND", "modal")
        monkeypatch.setenv("MATMUL_RUNNER", "gpu")
        from harness.tracing.telemetry import forward_gpu_env, inject_for_rollout

        class _E:
            id = "e1"
            split = "train"

        merged = inject_for_rollout(_E(), trace_id="t1", version=0, candidate="c0")
        assert merged["AR2_GPU_BACKEND"] == "modal"
        assert merged["MATMUL_RUNNER"] == "gpu"
        assert forward_gpu_env()["AR2_GPU_BACKEND"] == "modal"


class TestWriteSpan:
    def test_appends_json_line(self, tmp: Path, monkeypatch: pytest.MonkeyPatch):
        out = tmp / "trace.jsonl"
        monkeypatch.setenv("AR2_TRACE_ID", "tid1")
        monkeypatch.setenv("AR2_VERSION", "2")
        monkeypatch.setenv("AR2_CANDIDATE", "c9")
        monkeypatch.setenv("AR2_ENV_ID", "myenv")
        monkeypatch.setenv("AR2_SPLIT", "heldout")
        write_span(
            model="gpt-4o",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=88.0,
            tool_name="bash",
            tool_input="ls /",
            trace_file=out,
        )
        spans = parse_trace_file(out)
        assert len(spans) == 1
        s = spans[0]
        assert s["trace_id"] == "tid1"
        assert s["version"] == 2
        assert s["model"] == "gpt-4o"
        assert s["prompt_tokens"] == 10
        assert s["tool_name"] == "bash"

    def test_appends_multiple(self, tmp: Path, monkeypatch: pytest.MonkeyPatch):
        out = tmp / "trace.jsonl"
        monkeypatch.setenv("AR2_TRACE_ID", "t")
        monkeypatch.setenv("AR2_VERSION", "0")
        monkeypatch.setenv("AR2_CANDIDATE", "")
        monkeypatch.setenv("AR2_ENV_ID", "")
        monkeypatch.setenv("AR2_SPLIT", "train")
        for _ in range(3):
            write_span(model="o1", prompt_tokens=1, completion_tokens=1, latency_ms=1.0, trace_file=out)
        assert len(parse_trace_file(out)) == 3


class TestParseTraceFile:
    def test_empty_file(self, tmp: Path):
        f = tmp / "trace.jsonl"
        f.write_text("")
        assert parse_trace_file(f) == []

    def test_missing_file(self, tmp: Path):
        assert parse_trace_file(tmp / "missing.jsonl") == []

    def test_skips_malformed(self, tmp: Path):
        f = tmp / "trace.jsonl"
        f.write_text('{"a":1}\nnot-json\n{"b":2}\n')
        result = parse_trace_file(f)
        assert len(result) == 2
