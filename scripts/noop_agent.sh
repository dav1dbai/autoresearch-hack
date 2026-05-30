#!/usr/bin/env bash
# Minimal inner agent for GPU plumbing tests — appends a comment to kernel.py.
echo "# noop $(date +%s)" >> kernel.py
