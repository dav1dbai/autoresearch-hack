"""Offline tests for harness/backends/gpu.py — no real GPU instances."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGPUBackend:
    def _fresh_gpu_backend(self, monkeypatch):
        for key in list(sys.modules):
            if key in (
                "harness.backends.gpu",
                "harness.backends.local",
                "harness.backends.modal_gpu",
                "harness.backends.vast",
                "infra.modal.images",
                "infra.modal.secrets",
                "infra.vast.pool",
            ):
                monkeypatch.delitem(sys.modules, key, raising=False)
        monkeypatch.delenv("MODAL_ENVIRONMENT", raising=False)

    def test_local_backend_runs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MATMUL_STUB", "1")
        self._fresh_gpu_backend(monkeypatch)

        kernel = tmp_path / "kernel.py"
        kernel.write_text("import numpy as np\ndef matmul(A, B): return A @ B\n")

        from harness.backends.gpu import LocalBackend
        backend = LocalBackend()
        result = backend.run(kernel, {"M": 8, "N": 8, "K": 8, "reps": 3})
        assert isinstance(result, dict)
        assert "gflops" in result

    def test_make_gpu_backend_local(self, monkeypatch):
        self._fresh_gpu_backend(monkeypatch)
        from harness.backends.gpu import make_gpu_backend, LocalBackend
        assert isinstance(make_gpu_backend("local"), LocalBackend)

    def test_make_gpu_backend_modal(self, monkeypatch):
        self._fresh_gpu_backend(monkeypatch)
        from harness.backends.gpu import make_gpu_backend, ModalGPUBackend
        assert isinstance(make_gpu_backend("modal"), ModalGPUBackend)

    def test_make_gpu_backend_reads_env_after_import(self, monkeypatch):
        monkeypatch.delenv("AR2_GPU_BACKEND", raising=False)
        self._fresh_gpu_backend(monkeypatch)
        from harness.backends.gpu import make_gpu_backend, LocalBackend, ModalGPUBackend

        assert isinstance(make_gpu_backend(), LocalBackend)
        monkeypatch.setenv("AR2_GPU_BACKEND", "modal")
        assert isinstance(make_gpu_backend(), ModalGPUBackend)

    def test_make_gpu_backend_vast(self, monkeypatch):
        monkeypatch.setenv("AR2_BACKEND", "local")
        self._fresh_gpu_backend(monkeypatch)
        from harness.backends.gpu import make_gpu_backend, VastBackend
        assert isinstance(make_gpu_backend("vast"), VastBackend)

    def test_make_gpu_backend_invalid(self, monkeypatch):
        self._fresh_gpu_backend(monkeypatch)
        from harness.backends.gpu import make_gpu_backend
        with pytest.raises(ValueError, match="Unknown AR2_GPU_BACKEND"):
            make_gpu_backend("turbo")

    def test_modal_gpu_backend_checks_profile(self, monkeypatch):
        monkeypatch.setenv("MODAL_PROFILE", "wrong")
        self._fresh_gpu_backend(monkeypatch)
        for key in list(sys.modules):
            if key == "infra.modal.images":
                monkeypatch.delitem(sys.modules, key, raising=False)

        from harness.backends.gpu import ModalGPUBackend
        backend = ModalGPUBackend()
        with pytest.raises(RuntimeError, match="autoresearch-hack"):
            backend.run(Path("/tmp/fake.py"), {"M": 4, "N": 4, "K": 4, "reps": 1})

    def test_vast_backend_requires_instance_id(self, monkeypatch):
        monkeypatch.setenv("VAST_API_KEY", "fake-key")
        self._fresh_gpu_backend(monkeypatch)
        from harness.backends.gpu import VastBackend
        with pytest.raises(RuntimeError, match="instance_id"):
            VastBackend(instance_id=None).run(Path("/tmp/k.py"), {"M": 4})

    def test_vast_backend_requires_api_key(self, monkeypatch, tmp_path):
        self._fresh_gpu_backend(monkeypatch)
        monkeypatch.setenv("VAST_API_KEY", "")
        monkeypatch.delenv("MODAL_ENVIRONMENT", raising=False)
        from harness.backends.gpu import VastBackend

        kernel = tmp_path / "k.py"
        kernel.write_text("x")
        with pytest.raises(RuntimeError, match="VAST_API_KEY"):
            VastBackend(instance_id=42).run(kernel, {"M": 4})

    def test_make_gpu_backend_vast_with_modal_backend(self, monkeypatch):
        monkeypatch.setenv("AR2_BACKEND", "modal")
        monkeypatch.setenv("AR2_GPU_BACKEND", "vast")
        self._fresh_gpu_backend(monkeypatch)
        from harness.backends.gpu import VastBackend, make_gpu_backend
        assert isinstance(make_gpu_backend("vast"), VastBackend)

    def test_vast_backend_run_on_instance(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AR2_BACKEND", "local")
        monkeypatch.setenv("VAST_API_KEY", "fake-key")
        self._fresh_gpu_backend(monkeypatch)

        kernel = tmp_path / "kernel.py"
        kernel.write_text("import numpy as np\ndef matmul(A, B): return A @ B\n")
        eval_result = '{"gflops": 12.5, "correct": true, "seconds": 0.01}'

        with patch("infra.vast.pool.ssh_conn", return_value=("root@1.2.3.4", "12345")), \
             patch("infra.vast.pool.scp_to") as mock_scp, \
             patch("infra.vast.pool.ssh_run") as mock_ssh:
            mock_ssh.return_value = MagicMock(returncode=0, stdout=eval_result, stderr="")

            from harness.backends.gpu import VastBackend
            result = VastBackend(instance_id=99).run(
                kernel, {"M": 8, "N": 8, "K": 8, "reps": 3}
            )

        assert result["gflops"] == 12.5
        assert result["correct"] is True
        assert mock_scp.call_count == 2
        mock_ssh.assert_called_once()
        assert "eval_kernel.py" in mock_ssh.call_args.args[2]

    def test_gpu_backend_protocol(self, monkeypatch):
        self._fresh_gpu_backend(monkeypatch)
        from harness.backends.gpu import GPUBackend, LocalBackend
        assert isinstance(LocalBackend(), GPUBackend)
