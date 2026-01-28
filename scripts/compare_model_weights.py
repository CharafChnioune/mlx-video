#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import mlx.core as mx

from mlx_video.convert import sanitize_transformer_weights
from mlx_video.utils import get_model_path


def tensor_stats(arr: mx.array) -> dict:
    # Compute real stats for all dtypes (including uint32)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["dev", "distilled"], default="dev")
    parser.add_argument("--bf16-hf", default="Lightricks/LTX-2")
    parser.add_argument("--q8", required=True)
    parser.add_argument("--q4", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model_path = get_model_path(args.bf16_hf)
    bf16_file = model_path / f"ltx-2-19b-{args.pipeline}.safetensors"
    if not bf16_file.exists():
        raise FileNotFoundError(f"Missing bf16 weights: {bf16_file}")

    bf16_raw = mx.load(str(bf16_file))
    bf16 = sanitize_transformer_weights(bf16_raw)

    q8 = mx.load(str(Path(args.q8)))
    q4 = mx.load(str(Path(args.q4)))

    bf16_keys = set(bf16.keys())
    q8_keys = set(q8.keys())
    q4_keys = set(q4.keys())

    report = {
        "pipeline": args.pipeline,
        "bf16_file": str(bf16_file),
        "q8_file": str(Path(args.q8)),
        "q4_file": str(Path(args.q4)),
        "key_counts": {
            "bf16": len(bf16_keys),
            "q8": len(q8_keys),
            "q4": len(q4_keys),
        },
        "missing_in_q8": sorted(list(bf16_keys - q8_keys)),
        "missing_in_q4": sorted(list(bf16_keys - q4_keys)),
        "extra_in_q8": sorted(list(q8_keys - bf16_keys)),
        "extra_in_q4": sorted(list(q4_keys - bf16_keys)),
        "stats": {},
    }

    # Per-key stats for all shared keys (real values, no guessing)
    shared = sorted(list(bf16_keys & q8_keys & q4_keys))
    stats = {}
    for k in shared:
        stats[k] = {
            "bf16": tensor_stats(bf16[k]),
            "q8": tensor_stats(q8[k]),
            "q4": tensor_stats(q4[k]),
        }
    report["stats"] = stats

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote report: {out_path}")
    print(f"Shared keys: {len(shared)}")
    print(f"Missing in q8: {len(report['missing_in_q8'])} | Missing in q4: {len(report['missing_in_q4'])}")
    print(f"Extra in q8: {len(report['extra_in_q8'])} | Extra in q4: {len(report['extra_in_q4'])}")


if __name__ == "__main__":
    main()
