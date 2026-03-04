#!/usr/bin/env python3
"""Backward-compatible entrypoint for the tracking demo.

This file is kept for API stability and delegates to ``tracking_demo``.
"""

from tracking_demo import CONTROL_MODE, run_ui


if __name__ == "__main__":
    run_ui()
