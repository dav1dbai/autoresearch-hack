"""Tests for envs/base.py registry."""
from __future__ import annotations

from envs.base import _registry, list_envs, register
from envs.nanochat import NanoChatEnv


def test_register_and_list():
    before = len(_registry)
    env = NanoChatEnv(split="train", stub=True)
    register(env)
    assert len(list_envs()) == before + 1
    assert env in list_envs("train")
    assert env not in list_envs("heldout")
