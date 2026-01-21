#!/usr/bin/env python
"""Convenience wrapper for scripts/scale_match_check.py."""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_path("scripts/scale_match_check.py", run_name="__main__")
