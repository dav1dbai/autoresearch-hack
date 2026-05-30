"""Offline tests for Vast.ai pool helpers — mocked subprocess, no billing."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from infra.vast import pool as vast_pool


class TestParseSshUrl:
    def test_standard_format(self):
        assert vast_pool.parse_ssh_url("ssh root@1.2.3.4 -p 12345") == ("root@1.2.3.4", "12345")

    def test_port_before_host(self):
        assert vast_pool.parse_ssh_url("ssh -p 9999 root@host.example") == (
            "root@host.example",
            "9999",
        )

    def test_default_port(self):
        assert vast_pool.parse_ssh_url("ssh root@1.2.3.4") == ("root@1.2.3.4", "22")

    def test_quoted(self):
        assert vast_pool.parse_ssh_url('"ssh root@x -p 22"') == ("root@x", "22")

    def test_unparseable_raises(self):
        with pytest.raises(RuntimeError, match="Could not parse"):
            vast_pool.parse_ssh_url("not-ssh-output")


class TestRentGpu:
    @patch("infra.vast.pool._local_ssh_public_key", return_value=None)
    @patch("infra.vast.pool.wait_ssh")
    @patch("infra.vast.pool.run_vastai")
    @patch("infra.vast.pool.time.sleep")
    def test_rent_cheapest_offer_and_waits_ssh(self, _sleep, mock_run, mock_wait_ssh, _mock_pub):
        mock_run.side_effect = [
            json.dumps([{"id": 99, "dph_total": 0.40}, {"id": 100, "dph_total": 0.55}]),
            json.dumps({"new_contract": 42}),
            json.dumps({"actual_status": "running"}),
        ]
        mock_wait_ssh.return_value = ("root@host", "12345")

        inst = vast_pool.rent_gpu(api_key="test-key", gpu_name="RTX_4090", max_price=0.60)

        assert inst == 42
        mock_run.assert_any_call(
            ["search", "offers", "gpu_name=RTX_4090 num_gpus=1 rented=False dph<=0.6"],
            api_key="test-key",
        )
        mock_run.assert_any_call(
            [
                "create",
                "instance",
                "99",
                "--image",
                "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
                "--disk",
                "32",
                "--ssh",
            ],
            api_key="test-key",
        )
        mock_wait_ssh.assert_called_once_with(42, api_key="test-key", wait_s=pytest.approx(300, abs=15))

    @patch("infra.vast.pool.run_vastai")
    def test_rent_requires_api_key(self, mock_run, monkeypatch):
        monkeypatch.delenv("VAST_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="VAST_API_KEY"):
            vast_pool.rent_gpu()
        mock_run.assert_not_called()

    @patch("infra.vast.pool.run_vastai")
    def test_rent_no_offers(self, mock_run):
        mock_run.return_value = "[]"
        with pytest.raises(RuntimeError, match="No Vast offers"):
            vast_pool.rent_gpu(api_key="k", gpu_name="RTX_4090", max_price=0.60)


class TestDestroy:
    @patch("infra.vast.pool.subprocess.run")
    def test_destroy_calls_vastai(self, mock_run, monkeypatch):
        monkeypatch.setenv("VASTAI_BIN", "/fake/vastai")
        vast_pool.destroy(7, api_key="secret")
        mock_run.assert_called_once_with(
            ["/fake/vastai", "destroy", "instance", "7", "--api-key", "secret"],
            check=False,
            capture_output=True,
        )

    @patch("infra.vast.pool.subprocess.run")
    def test_destroy_skips_without_api_key(self, mock_run, monkeypatch):
        monkeypatch.delenv("VAST_API_KEY", raising=False)
        vast_pool.destroy(7, api_key="")
        mock_run.assert_not_called()


class TestWaitSsh:
    @patch("infra.vast.pool.ssh_run")
    @patch("infra.vast.pool.ssh_conn")
    @patch("infra.vast.pool.time.sleep")
    @patch("infra.vast.pool.time.time")
    def test_wait_ssh_retries_until_ok(self, mock_time, _sleep, mock_conn, mock_ssh_run):
        mock_time.side_effect = [0, 0, 11, 11]
        mock_conn.return_value = ("root@host", "2222")
        fail = MagicMock(returncode=1, stdout="", stderr="connection refused")
        ok = MagicMock(returncode=0, stdout="ok\n", stderr="")
        mock_ssh_run.side_effect = [fail, ok]

        conn = vast_pool.wait_ssh(5, api_key="k", wait_s=30, interval=10)

        assert conn == ("root@host", "2222")
        assert mock_ssh_run.call_count == 2

    @patch("infra.vast.pool.ssh_conn")
    @patch("infra.vast.pool.time.sleep")
    @patch("infra.vast.pool.time.time")
    def test_wait_ssh_timeout(self, mock_time, _sleep, mock_conn):
        mock_time.side_effect = [0, 31]
        mock_conn.side_effect = RuntimeError("ssh-url failed")

        with pytest.raises(RuntimeError, match="SSH not ready"):
            vast_pool.wait_ssh(5, api_key="k", wait_s=30, interval=10)


class TestBootstrapInstance:
    @patch("infra.vast.pool.ssh_run")
    @patch("infra.vast.pool.ssh_conn")
    def test_bootstrap_skips_when_deps_present(self, mock_conn, mock_ssh_run):
        mock_conn.return_value = ("root@h", "22")
        mock_ssh_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")

        vast_pool.bootstrap_instance(1, api_key="k")

        mock_ssh_run.assert_called_once()

    @patch("infra.vast.pool.ssh_run")
    @patch("infra.vast.pool.ssh_conn")
    def test_bootstrap_pip_installs_when_missing(self, mock_conn, mock_ssh_run):
        mock_conn.return_value = ("root@h", "22")
        missing = MagicMock(returncode=1, stdout="", stderr="ModuleNotFoundError")
        installed = MagicMock(returncode=0, stdout="ok\n", stderr="")
        mock_ssh_run.side_effect = [missing, installed]

        vast_pool.bootstrap_instance(1, api_key="k")

        assert mock_ssh_run.call_count == 2
        assert "pip install" in mock_ssh_run.call_args_list[1].args[2]

    @patch("infra.vast.pool.ssh_run")
    @patch("infra.vast.pool.ssh_conn")
    def test_bootstrap_raises_on_pip_failure(self, mock_conn, mock_ssh_run):
        mock_conn.return_value = ("root@h", "22")
        missing = MagicMock(returncode=1, stdout="", stderr="no numpy")
        failed = MagicMock(returncode=1, stdout="", stderr="pip error")
        mock_ssh_run.side_effect = [missing, failed]

        with pytest.raises(RuntimeError, match="bootstrap_instance failed"):
            vast_pool.bootstrap_instance(1, api_key="k")


class TestScpTo:
    @patch("infra.vast.pool.subprocess.check_call")
    @patch("infra.vast.pool.time.sleep")
    def test_scp_retries_then_succeeds(self, _sleep, mock_call):
        mock_call.side_effect = [OSError("fail"), None]
        vast_pool.scp_to("/local.py", "root@h", "22", "/remote.py", retries=3)
        assert mock_call.call_count == 2

    @patch("infra.vast.pool.subprocess.check_call")
    @patch("infra.vast.pool.time.sleep")
    def test_scp_raises_after_retries(self, _sleep, mock_call):
        import subprocess

        mock_call.side_effect = subprocess.CalledProcessError(1, "scp")
        with pytest.raises(RuntimeError, match="scp failed"):
            vast_pool.scp_to("/a", "root@h", "22", "/b", retries=2)


class TestParseSshUrl:
    def test_ssh_uri_with_port(self):
        assert vast_pool.parse_ssh_url('ssh://root@ssh7.vast.ai:33790') == ('root@ssh7.vast.ai', '33790')
