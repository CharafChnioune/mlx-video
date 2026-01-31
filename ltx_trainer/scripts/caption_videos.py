#!/usr/bin/env python3
"""MLX captioner wrapper for videos/images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_video.mlx_trainer.captioning import get_captioner
from mlx_video.mlx_trainer.video_utils import read_video

SUPPORTED = {".mp4", ".mov", ".mkv", ".avi", ".png", ".jpg", ".jpeg"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--backend", default="mlx_vlm")
    parser.add_argument("--model-id", default="mlx-community/SmolVLM-Instruct-4bit")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in SUPPORTED]
    if not files:
        raise SystemExit("No media files found.")

    captioner = get_captioner(args.backend, args.model_id, args.max_new_tokens)
    out = {}
    for path in files:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            from PIL import Image
            frame = Image.open(path).convert("RGB")
            import numpy as np
            frame_np = np.array(frame)
            caption = captioner.caption(frame_np)
        else:
            frames, _fps = read_video(str(path), max_frames=1)
            frame_np = frames[0]
            caption = captioner.caption(frame_np)
        out[path.name] = caption
        if args.debug:
            print(f"[caption] {path.name}: {caption}")

    Path(args.output_json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
