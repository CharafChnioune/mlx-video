#!/usr/bin/env python3
"""Scene split wrapper.

If scenedetect is installed, this will run it; otherwise it provides a clear
error for MLX-only environments.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=30.0)
    args = parser.parse_args()

    cmd = [
        "scenedetect",
        "-i",
        args.input,
        "-o",
        args.output_dir,
        "detect-content",
        "-t",
        str(args.threshold),
        "split-video",
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit("scenedetect not installed. Install py-scenedetect to use this script.")


if __name__ == "__main__":
    main()
