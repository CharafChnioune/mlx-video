import argparse
import json
from pathlib import Path

import mlx.core as mx

from mlx_video.models.ltx.text_encoder import LTX2TextEncoder
from mlx_video.utils import get_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance a prompt using Gemma (MLX).")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Prompt to enhance.")
    parser.add_argument(
        "--enhancer-repo",
        type=str,
        default=None,
        help="Optional enhancer model repo. Overrides --mode.",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="Lightricks/LTX-2",
        help="Base model repo (for system prompt + connectors).",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for enhancement.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional custom system prompt string.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=None,
        help="Optional path to a custom system prompt file.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = get_model_path(args.model_repo)

    enhancer_repo = args.enhancer_repo
    # Default: use the model's bundled text_encoder if present.
    if enhancer_repo:
        text_encoder_path = enhancer_repo
    else:
        bundled = Path(model_path) / "text_encoder"
        text_encoder_path = str(bundled) if bundled.is_dir() else str(model_path)

    enhancer = LTX2TextEncoder()
    enhancer.load(model_path=model_path, text_encoder_path=text_encoder_path)
    mx.eval(enhancer.parameters())

    system_prompt = None
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text()
    elif args.system_prompt:
        system_prompt = args.system_prompt

    enhanced = enhancer.enhance_t2v(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        verbose=False,
        system_prompt=system_prompt,
    )

    if args.json:
        print(json.dumps({"enhanced": enhanced}))
    else:
        print(enhanced)


if __name__ == "__main__":
    main()
