"""Register all Modal functions on infra.modal.images.app."""
from __future__ import annotations

from infra.modal.images import app  # noqa: F401 — shared app registry


def register_all() -> None:
    """Side-effect imports attach @app.function handlers to the shared app."""
    import harness.cloud.runner  # noqa: F401
    import harness.backends.modal_gpu  # noqa: F401
    import harness.backends.vast  # noqa: F401
