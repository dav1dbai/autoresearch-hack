"""Modal workspace secrets — names + host-side sync from .env.

Vast scoring runs in `_vast_gpu_run` on Modal; the API key must live in a Modal
secret (not in agent container env). Create once via CLI:

    modal secret create autoresearch-vast VAST_API_KEY=your_key --force

Or let the harness sync it on `--vast-rent` (reads VAST_API_KEY from .env).
"""
from __future__ import annotations

import os
import subprocess

import modal

VAST_SECRET_NAME = os.environ.get("AR2_VAST_SECRET", "autoresearch-vast")
OPENAI_SECRET_NAME = "autoresearch-openai"


def vast_secret() -> modal.Secret:
    return modal.Secret.from_name(VAST_SECRET_NAME)


def ensure_vast_modal_secret(*, api_key: str | None = None) -> str:
    """Create/update the Vast Modal secret from the host environment."""
    from infra.modal.images import assert_hackathon_profile

    key = api_key or os.environ.get("VAST_API_KEY", "")
    if not key:
        raise RuntimeError(
            "VAST_API_KEY not set — add to .env before --vast-rent or run:\n"
            f"  modal secret create {VAST_SECRET_NAME} VAST_API_KEY=... --force"
        )
    assert_hackathon_profile()
    subprocess.run(
        [
            "modal", "secret", "create", VAST_SECRET_NAME,
            f"VAST_API_KEY={key}",
            "--force",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return VAST_SECRET_NAME
