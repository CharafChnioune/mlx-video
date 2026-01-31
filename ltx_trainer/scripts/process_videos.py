#!/usr/bin/env python3
"""MLX-native replacement for LTX-2 process_videos.py.

This script maps common LTX-2 dataset formats (CSV/JSON/JSONL or folder) into
MLX precompute latents via mlx_video.mlx_trainer.precompute.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from mlx_video.mlx_trainer import precompute


SUPPORTED_MEDIA = {".mp4", ".mov", ".mkv", ".avi", ".png", ".jpg", ".jpeg"}


def _read_records(path: Path) -> list[dict]:
    if path.is_dir():
        return [{"path": str(p)} for p in sorted(path.iterdir()) if p.suffix.lower() in SUPPORTED_MEDIA]
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="") as f:
            return list(csv.DictReader(f))
    if path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            out = []
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
            return out
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # allow {"records": [...]} format
            for key in ("records", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        raise ValueError(f"Unsupported JSON structure in {path}")
    raise ValueError(f"Unsupported dataset format: {path}")


def _pick_key(record: dict, keys: Iterable[str]) -> str | None:
    for key in keys:
        if key in record and record[key]:
            return key
    return None


def _resolve_media_path(raw: str, base: Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _clean_prompt(prompt: str) -> str:
    text = prompt.strip()
    prefixes = [
        "caption:",
        "the image shows",
        "this image shows",
        "the video shows",
        "this video shows",
        "image of",
        "video of",
        "a video of",
        "a photo of",
        "an image of",
    ]
    lowered = text.lower()
    for pref in prefixes:
        if lowered.startswith(pref):
            text = text[len(pref) :].lstrip(" :,-")
            break
    return text


def _build_temp_input_dir(
    records: list[dict],
    base_dir: Path,
    video_key: str,
    prompt_key: str | None,
    reference_key: str | None,
    lora_trigger: str | None,
    remove_llm_prefixes: bool,
) -> tuple[Path, Path | None, Path | None]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="mlx_precompute_"))
    ref_dir = None
    if reference_key:
        ref_dir = Path(tempfile.mkdtemp(prefix="mlx_precompute_ref_"))
    prompts = {}
    for idx, rec in enumerate(records):
        raw = rec.get(video_key)
        if not raw:
            continue
        src = _resolve_media_path(str(raw), base_dir)
        if not src.exists():
            continue
        name = f"{idx:05d}_{src.name}"
        dest = tmp_dir / name
        try:
            dest.symlink_to(src)
        except Exception:
            dest.write_bytes(src.read_bytes())

        if reference_key and rec.get(reference_key) and ref_dir is not None:
            ref_src = _resolve_media_path(str(rec.get(reference_key)), base_dir)
            if ref_src.exists():
                ref_dest = ref_dir / name
                try:
                    ref_dest.symlink_to(ref_src)
                except Exception:
                    ref_dest.write_bytes(ref_src.read_bytes())

        if prompt_key and rec.get(prompt_key):
            prompt = str(rec.get(prompt_key))
            if remove_llm_prefixes:
                prompt = _clean_prompt(prompt)
            if lora_trigger:
                prompt = f"{lora_trigger}, {prompt}"
            prompts[name] = prompt
    prompts_path = None
    if prompts:
        prompts_path = tmp_dir / "prompts.json"
        prompts_path.write_text(json.dumps(prompts, indent=2))
    return tmp_dir, prompts_path, ref_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX precompute wrapper for LTX-2 datasets")
    parser.add_argument("dataset", help="Dataset file (csv/json/jsonl) or folder")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-source", default="Lightricks/LTX-2")
    parser.add_argument("--text-encoder-repo", default=None)
    parser.add_argument("--caption", action="store_true")
    parser.add_argument("--caption-model", default="mlx-community/SmolVLM-Instruct-4bit")
    parser.add_argument("--caption-backend", default="mlx_vlm")
    parser.add_argument("--prompt-column", default=None)
    parser.add_argument("--caption-column", default=None)
    parser.add_argument("--video-column", default=None)
    parser.add_argument("--media-column", default=None)
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--reference-column", default=None)
    parser.add_argument("--lora-trigger", default=None)
    parser.add_argument("--remove-llm-prefixes", action="store_true")
    parser.add_argument("--audio-latents-dir", default=None)
    parser.add_argument("--resolution-buckets", default=None, help="e.g. 832x480x73;768x768x65")
    parser.add_argument("--with-audio", action="store_true")
    parser.add_argument("--frame-cap", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    ref_dir = None
    if dataset_path.is_dir():
        input_dir = dataset_path
        prompts_file = None
    else:
        records = _read_records(dataset_path)
        if not records:
            raise SystemExit("No records found.")
        video_key = (
            args.video_column
            or args.media_column
            or _pick_key(records[0], ["video", "path", "file", "video_path", "media"])
            or "video"
        )
        prompt_key = (
            args.prompt_column
            or args.caption_column
            or _pick_key(records[0], ["prompt", "caption", "text"])
        )
        input_dir, prompts_file, ref_dir = _build_temp_input_dir(
            records,
            dataset_path.parent,
            video_key,
            prompt_key,
            args.reference_column,
            args.lora_trigger,
            args.remove_llm_prefixes,
        )

    argv = [
        "precompute",
        "--input-dir", str(input_dir),
        "--output-dir", str(Path(args.output_dir).expanduser().resolve()),
        "--model-repo", args.model_source,
    ]
    if args.text_encoder_repo:
        argv += ["--text-encoder-repo", args.text_encoder_repo]
    if prompts_file:
        argv += ["--prompts-file", str(prompts_file)]
    if args.caption:
        argv += ["--caption", "--caption-model", args.caption_model, "--caption-backend", args.caption_backend]
    if args.reference_dir:
        argv += ["--reference-dir", args.reference_dir]
    elif ref_dir is not None:
        argv += ["--reference-dir", str(ref_dir)]
    if args.audio_latents_dir:
        argv += ["--audio-latents-dir", args.audio_latents_dir]
    if args.with_audio:
        argv += ["--with-audio"]
    if args.resolution_buckets:
        argv += ["--resolution-buckets", args.resolution_buckets]
    if args.frame_cap:
        argv += ["--frame-cap", str(args.frame_cap)]
    if args.debug:
        argv += ["--debug"]

    if args.debug:
        print("[process_videos] invoking precompute:", " ".join(argv))

    sys.argv = argv
    precompute.main()


if __name__ == "__main__":
    main()
