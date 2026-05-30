"""OpenAI-compatible streaming benchmark client."""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx


@dataclass
class RequestMetrics:
    ttft_ms: float
    itl_ms: list[float]
    output_tokens: int
    e2e_ms: float
    error: str | None = None

    @property
    def mean_itl_ms(self) -> float | None:
        return statistics.mean(self.itl_ms) if self.itl_ms else None


def _percentiles(values: list[float], ps: tuple[int, ...] = (50, 95)) -> dict[str, float]:
    if not values:
        return {f"p{p}": 0.0 for p in ps}
    s = sorted(values)
    out: dict[str, float] = {}
    for p in ps:
        idx = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
        out[f"p{p}"] = round(s[idx], 4)
    return out


def aggregate(metrics: list[RequestMetrics]) -> dict[str, Any]:
    ok = [m for m in metrics if not m.error]
    err = [m for m in metrics if m.error]
    if not ok:
        return {"n_ok": 0, "n_error": len(err), "errors": [m.error for m in err[:5]]}

    ttfts = [m.ttft_ms for m in ok]
    e2es = [m.e2e_ms for m in ok]
    itls = [x for m in ok for x in m.itl_ms]
    out_tokens = sum(m.output_tokens for m in ok)
    wall = sum(m.e2e_ms for m in ok) / 1000.0
    return {
        "n_ok": len(ok),
        "n_error": len(err),
        "ttft_ms": _percentiles(ttfts),
        "itl_ms": _percentiles(itls) if itls else {},
        "e2e_ms": _percentiles(e2es),
        "output_tokens": out_tokens,
        "output_tok_s": round(out_tokens / wall, 4) if wall > 0 else 0.0,
        "mean_itl_ms": round(statistics.mean(itls), 4) if itls else None,
    }


async def _one_request(
    client: httpx.AsyncClient,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> RequestMetrics:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    itl_ms: list[float] = []
    output_tokens = 0
    last_t = t0
    try:
        async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                now = time.perf_counter()
                if ttft_ms is None:
                    ttft_ms = (now - t0) * 1000
                else:
                    itl_ms.append((now - last_t) * 1000)
                last_t = now
                output_tokens += 1
    except Exception as ex:
        return RequestMetrics(0, [], 0, (time.perf_counter() - t0) * 1000, error=str(ex)[:300])

    e2e = (time.perf_counter() - t0) * 1000
    return RequestMetrics(ttft_ms or e2e, itl_ms, output_tokens, e2e)


async def run_workload(
    base_url: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    concurrency: int,
    n_requests: int,
    warmup: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestMetrics] = []

    async def worker(client: httpx.AsyncClient) -> None:
        async with sem:
            results.append(
                await _one_request(
                    client,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )

    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s, limits=limits) as client:
        total = warmup + n_requests
        await asyncio.gather(*[worker(client) for _ in range(total)])
        measured = results[warmup:]
    return aggregate(measured)


def run_workload_sync(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(run_workload(**kwargs))


def metrics_to_json(metrics: list[RequestMetrics]) -> list[dict[str, Any]]:
    return [asdict(m) for m in metrics]
