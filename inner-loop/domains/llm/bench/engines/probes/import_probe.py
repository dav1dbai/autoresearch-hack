#!/usr/bin/env python3
"""Probe Python package import for minimal-stack engines."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--import-name", required=True)
    args = p.parse_args()
    try:
        mod = __import__(args.import_name)
        print(json.dumps({"ok": True, "version": getattr(mod, "__version__", "unknown")}))
        return 0
    except Exception as ex:
        print(json.dumps({"ok": False, "error": str(ex)[:400]}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
