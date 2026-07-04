#!/usr/bin/env python
"""Thin wrapper for the omni chat CLI; see `python scripts/chat.py --help`."""

from omni.infer.chat import main

if __name__ == "__main__":
    raise SystemExit(main())
