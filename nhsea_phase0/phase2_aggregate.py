#!/usr/bin/env python
"""Convenience wrapper for scripts/phase2_aggregate.py."""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_path("scripts/phase2_aggregate.py", run_name="__main__")
