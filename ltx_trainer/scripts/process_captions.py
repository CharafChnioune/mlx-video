#!/usr/bin/env python3
"""Convert captions JSON to prompts file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.input_json).read_text())
    lines = []
    for name, prompt in data.items():
        lines.append(f"{name}|{prompt}")
    Path(args.output).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
