"""Rent and release Vast.ai GPU instances for kernel benchmarks.

TODO (SSH auth — blocked on team API keys):
  Team Vast accounts (`is_team: true`) cannot register SSH keys (`vastai show ssh-keys`
  → []). Rent succeeds but `wait_ssh` fails with Permission denied (publickey) on the
  Vast SSH gateway; `SSH_PUBLIC_KEY` at create time does not fix team-key auth.

  Unblock options:
    - Use a personal VAST_API_KEY with an SSH public key on that account, or
    - Team admin attaches keys on the parent account; wire post-rent
      `vastai attach ssh <instance_id> <key_id>` when VAST_SSH_KEY_ID is set.

  Live failure: instance 38636988, rent OK, wait_ssh failed (2026-05-30).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]


def _ssh_identity_args() -> list[str]:
    ident = os.environ.get("VAST_SSH_IDENTITY", str(Path.home() / ".ssh" / "id_ed25519"))
    path = Path(ident).expanduser()
    if path.is_file():
        return ["-i", str(path)]
    return []


def _local_ssh_public_key() -> str | None:
    override = os.environ.get("VAST_SSH_PUBLIC_KEY", "").strip()
    if override:
        return override
    pub = Path.home() / ".ssh" / "id_ed25519.pub"
    if pub.is_file():
        return pub.read_text().strip()
    return None


def vastai_bin() -> Path:
    """Path to the vastai CLI (override with VASTAI_BIN)."""
    override = os.environ.get("VASTAI_BIN")
    if override:
        return Path(override)
    import shutil
    found = shutil.which("vastai")
    if found:
        return Path(found)
    return Path(".venv/bin/vastai")


def run_vastai(args: list[str], *, api_key: str, raw: bool = True) -> str:
    cmd = [str(vastai_bin()), *args, "--api-key", api_key]
    if raw:
        cmd.append("--raw")
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    return out.strip()


def ssh_url(instance_id: int, *, api_key: str | None = None) -> str:
    key = api_key or os.environ.get("VAST_API_KEY", "")
    return run_vastai(["ssh-url", str(instance_id)], api_key=key)


def parse_ssh_url(raw: str) -> tuple[str, str]:
    """Parse vastai ssh-url / scp-url output -> (user@host, port)."""
    text = raw.strip().strip('"').strip("'")
    for prefix in ("ssh://", "scp://"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            if "@" in text and text.rsplit(":", 1)[-1].isdigit():
                user_host, port = text.rsplit(":", 1)
                return user_host, port
            raise RuntimeError(f"Could not parse ssh-url: {raw!r}")
    if text.startswith("ssh "):
        text = text[4:]
    parts = text.split()
    user_host: str | None = None
    port = "22"
    i = 0
    while i < len(parts):
        if parts[i] == "-p" and i + 1 < len(parts):
            port = parts[i + 1]
            i += 2
        elif "@" in parts[i]:
            user_host = parts[i]
            i += 1
        else:
            i += 1
    if not user_host:
        raise RuntimeError(f"Could not parse ssh-url: {raw!r}")
    return user_host, port


def ssh_conn(instance_id: int, *, api_key: str | None = None) -> tuple[str, str]:
    """Return (user@host, port) for a Vast instance."""
    return parse_ssh_url(ssh_url(instance_id, api_key=api_key))


def ssh_run(
    user_host: str,
    port: str,
    remote_cmd: str,
    *,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-p", port, *SSH_OPTS, *_ssh_identity_args(), user_host, remote_cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def scp_to(
    local_path: str,
    user_host: str,
    port: str,
    remote_path: str,
    *,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> None:
    cmd = [
        "scp", "-P", port, "-o", "StrictHostKeyChecking=no",
        *_ssh_identity_args(),
        local_path, f"{user_host}:{remote_path}",
    ]
    last_err = ""
    for attempt in range(retries):
        try:
            subprocess.check_call(cmd, timeout=120)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            last_err = str(e)
        if attempt < retries - 1:
            time.sleep(retry_delay)
    raise RuntimeError(f"scp failed after {retries} attempts: {last_err}")


def wait_ssh(
    instance_id: int,
    *,
    api_key: str | None = None,
    wait_s: int = 300,
    interval: float = 10.0,
) -> tuple[str, str]:
    """Wait until SSH accepts connections; return (user@host, port).

    See module TODO if this fails with Permission denied (publickey) on team API keys.
    """
    key = api_key or os.environ.get("VAST_API_KEY", "")
    deadline = time.time() + wait_s
    last_err = ""
    while time.time() < deadline:
        try:
            user_host, port = ssh_conn(instance_id, api_key=key)
            result = ssh_run(user_host, port, "echo ok", timeout=15)
            if result.returncode == 0 and "ok" in result.stdout:
                return user_host, port
            last_err = (result.stderr or result.stdout or "nonzero exit").strip()
        except Exception as e:
            last_err = str(e)
        time.sleep(interval)
    raise RuntimeError(
        f"SSH not ready for Vast instance {instance_id} after {wait_s}s: {last_err}"
    )


def bootstrap_instance(
    instance_id: int,
    *,
    api_key: str | None = None,
    conn: tuple[str, str] | None = None,
) -> None:
    """Ensure numpy and torch are available on the remote instance."""
    key = api_key or os.environ.get("VAST_API_KEY", "")
    user_host, port = conn or ssh_conn(instance_id, api_key=key)

    check = ssh_run(
        user_host,
        port,
        "python3 -c 'import numpy; import torch; print(\"ok\")'",
        timeout=60,
    )
    if check.returncode == 0 and "ok" in check.stdout:
        return

    install = ssh_run(
        user_host,
        port,
        "python3 -m pip install -q numpy torch",
        timeout=600,
    )
    if install.returncode != 0:
        err = (install.stderr or install.stdout or "pip install failed").strip()
        raise RuntimeError(f"bootstrap_instance failed on {instance_id}: {err}")


def rent_gpu(
    *,
    api_key: str | None = None,
    gpu_name: str | None = None,
    min_gflops_target: float | None = None,
    max_price: float | None = None,
    wait_s: int = 300,
) -> int:
    """Search offers, create an instance, wait until running + SSH-ready; return id."""
    key = api_key or os.environ.get("VAST_API_KEY", "")
    if not key:
        raise RuntimeError("VAST_API_KEY not set")

    gpu = gpu_name or os.environ.get("VAST_GPU_NAME", "RTX_4090")
    query = f"gpu_name={gpu} num_gpus=1 rented=False"
    if max_price is None:
        max_price = float(os.environ.get("VAST_MAX_PRICE", "0.60"))
    query += f" dph<={max_price}"

    offers_raw = run_vastai(["search", "offers", query], api_key=key)
    offers = json.loads(offers_raw) if offers_raw.startswith("[") else []
    if not offers:
        fallback_price = float(os.environ.get("VAST_MAX_PRICE_FALLBACK", str(max_price * 2)))
        fallback_query = f"num_gpus=1 rented=False dph<={fallback_price}"
        offers_raw = run_vastai(["search", "offers", fallback_query], api_key=key)
        offers = json.loads(offers_raw) if offers_raw.startswith("[") else []
        query = fallback_query
    if not offers:
        raise RuntimeError(f"No Vast offers for query: {query}")

    offer = sorted(offers, key=lambda o: float(o.get("dph_total", 999)))[0]
    offer_id = offer["id"]
    image = os.environ.get(
        "VAST_IMAGE",
        "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
    )
    disk = os.environ.get("VAST_DISK_GB", "32")
    create_args = [
        "create",
        "instance",
        str(offer_id),
        "--image",
        image,
        "--disk",
        disk,
        "--ssh",
    ]
    pub = _local_ssh_public_key()
    if pub:
        create_args.extend(["--env", f"-e SSH_PUBLIC_KEY={pub}"])
    create_raw = run_vastai(create_args, api_key=key)
    created = json.loads(create_raw) if create_raw.startswith("{") else {}
    instance_id = int(created.get("new_contract") or created.get("id") or 0)
    if not instance_id:
        raise RuntimeError(f"vastai create instance failed: {create_raw[:300]}")

    deadline = time.time() + wait_s
    while time.time() < deadline:
        inst_raw = run_vastai(["show", "instance", str(instance_id)], api_key=key)
        inst = json.loads(inst_raw) if inst_raw.startswith("{") else {}
        status = str(inst.get("actual_status") or inst.get("status") or "")
        if status.lower() in ("running", "success"):
            wait_ssh(instance_id, api_key=key, wait_s=max(0, int(deadline - time.time())))
            return instance_id
        time.sleep(10)

    raise RuntimeError(f"Vast instance {instance_id} not running after {wait_s}s")


def destroy(instance_id: int, *, api_key: str | None = None) -> None:
    key = api_key or os.environ.get("VAST_API_KEY", "")
    if not key:
        return
    subprocess.run(
        [str(vastai_bin()), "destroy", "instance", str(instance_id), "--api-key", key],
        check=False,
        capture_output=True,
    )
