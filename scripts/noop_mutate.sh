#!/usr/bin/env bash
# Minimal meta-agent for outer-loop plumbing — touch entrypoint.py only.
echo "# noop-mutate $(date +%s)" >> entrypoint.py
