#!/usr/bin/env python3
"""Compute edge-map reference media for IC-LoRA conditioning (MLX-only).

Supports:
- input directory of media files
- dataset CSV/JSON/JSONL with media column (adds reference_path column)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mlx_video.mlx_trainer.video_utils import read_video, save_video

SUPPORTED = {".mp4", ".mov", ".mkv", ".avi", ".png", ".jpg", ".jpeg"}


def _edges(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = np.stack([edges] * 3, axis=-1)
    return edges


def _read_dataset(path: Path) -> tuple[list[dict], str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="") as f:
            return list(csv.DictReader(f)), "csv"
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list of records")
        return data, "json"
    if path.suffix.lower() == ".jsonl":
        out = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out, "jsonl"
    raise ValueError(f"Unsupported dataset format: {path}")


def _write_dataset(path: Path, records: list[dict], fmt: str) -> None:
    if fmt == "csv":
        if not records:
            return
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        return
    if fmt == "json":
        path.write_text(json.dumps(records, indent=2))
        return
    if fmt == "jsonl":
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records))
        return
    raise ValueError(f"Unsupported format: {fmt}")


def _compute_reference_for_media(path: Path, output_path: Path, max_frames: int, override: bool, debug: bool) -> None:
    if output_path.exists() and not override:
        if debug:
            print(f"[reference] skip existing {output_path}")
        return
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        frame = np.array(Image.open(path).convert("RGB"))
        edge = _edges(frame)
        Image.fromarray(edge).save(output_path)
        if debug:
            print(f"[reference] {path.name} -> image")
        return
    frames, fps = read_video(str(path), max_frames=max_frames)
    edge_frames = np.stack([_edges(f) for f in frames], axis=0)
    save_video(edge_frames, str(output_path), fps=fps)
    if debug:
        print(f"[reference] {path.name} -> {output_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--output", default=None, help="Output dataset file (defaults to dataset-file)")
    parser.add_argument("--media-column", default="media_path")
    parser.add_argument("--reference-column", default="reference_path")
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.dataset_file:
        dataset_path = Path(args.dataset_file).expanduser().resolve()
        records, fmt = _read_dataset(dataset_path)
        base_dir = dataset_path.parent
        for rec in records:
            media = rec.get(args.media_column)
            if not media:
                continue
            media_path = (base_dir / str(media)).resolve()
            if not media_path.exists():
                if args.debug:
                    print(f"[reference] missing media: {media_path}")
                continue
            ref_path = media_path.parent / f"{media_path.stem}_reference{media_path.suffix}"
            _compute_reference_for_media(media_path, ref_path, args.max_frames, args.override, args.debug)
            rec[args.reference_column] = str(ref_path.relative_to(base_dir))
        out_path = Path(args.output).expanduser().resolve() if args.output else dataset_path
        _write_dataset(out_path, records, fmt)
        if args.debug:
            print(f"[reference] wrote dataset with references: {out_path}")
        return

    if not args.input_dir or not args.output_dir:
        raise SystemExit("--input-dir/--output-dir or --dataset-file is required.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in SUPPORTED]
    if not files:
        raise SystemExit("No media files found.")

    for path in files:
        out = output_dir / f"{path.stem}_reference{path.suffix}"
        _compute_reference_for_media(path, out, args.max_frames, args.override, args.debug)


if __name__ == "__main__":
    main()
