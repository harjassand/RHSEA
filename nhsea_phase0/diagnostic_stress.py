#!/usr/bin/env python
"""Convenience wrapper for scripts/diagnostic_stress.py."""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_path("scripts/diagnostic_stress.py", run_name="__main__")
