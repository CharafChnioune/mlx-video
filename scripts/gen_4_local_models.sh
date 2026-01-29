#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
CONVERTED="$REPO_ROOT/converted"
OUT="$REPO_ROOT/outputvideos"

PROMPT="${PROMPT:-A cinematic panorama of a mountain landscape at sunrise, mist in the valleys, soft sun rays, birds singing in the background, realistic lighting, filmic look}"
TEXT_ENCODER_REPO="${TEXT_ENCODER_REPO:-bobwu/gemma-3-12b-it-abliterated-mlx-4Bit}"
# Use the original LTX-2 text encoder for quant tests to avoid mismatch
TEXT_ENCODER_REPO_Q="${TEXT_ENCODER_REPO_Q:-Lightricks/LTX-2}"

DEV_8BIT="$CONVERTED/ltx2-dev-8bit-mlx"
DEV_4BIT="$CONVERTED/ltx2-dev-4bit-mlx"
DEV_2BIT="$CONVERTED/ltx2-dev-2bit-mlx"
DIS_8BIT="$CONVERTED/ltx2-distilled-8bit-mlx"
DIS_4BIT="$CONVERTED/ltx2-distilled-4bit-mlx"
DIS_2BIT="$CONVERTED/ltx2-distilled-2bit-mlx"

mkdir -p "$OUT"

echo "==> Generating BF16 + 4 local-model videos"

./.venv/bin/python -m mlx_video.generate \
  --prompt "$PROMPT" \
  --pipeline dev \
  --width 832 --height 480 \
  --fps 24 --num-frames 73 \
  --steps 20 --cfg-scale 4 \
  --seed 42 \
  --audio \
  --output-path "$OUT/test_dev_bf16_audio_3s.mp4"

./.venv/bin/python -m mlx_video.generate \
  --prompt "$PROMPT" \
  --pipeline distilled \
  --width 832 --height 512 \
  --fps 24 --num-frames 73 \
  --steps 8 --cfg-scale 4 \
  --seed 42 \
  --audio \
  --output-path "$OUT/test_distilled_bf16_audio_3s.mp4"

./.venv/bin/python -m mlx_video.generate \
  --prompt "$PROMPT" \
  --pipeline dev \
  --model-repo "$DEV_8BIT" \
  --text-encoder-repo "$TEXT_ENCODER_REPO_Q" \
  --width 832 --height 480 \
  --fps 24 --num-frames 73 \
  --steps 20 --cfg-scale 4 \
  --seed 42 \
  --audio \
  --output-path "$OUT/test_dev_8bit_audio_5s.mp4"

./.venv/bin/python -m mlx_video.generate \
  --prompt "$PROMPT" \
  --pipeline dev \
  --model-repo "$DEV_4BIT" \
  --text-encoder-repo "$TEXT_ENCODER_REPO_Q" \
  --width 832 --height 480 \
  --fps 24 --num-frames 73 \
  --steps 20 --cfg-scale 4 \
  --seed 42 \
  --audio \
  --output-path "$OUT/test_dev_4bit_audio_5s.mp4"

./.venv/bin/python -m mlx_video.generate \
  --prompt "$PROMPT" \
  --pipeline dev \
  --model-repo "$DEV_2BIT" \
  --text-encoder-repo "$TEXT_ENCODER_REPO_Q" \
  --width 832 --height 480 \
  --fps 24 --num-frames 73 \
  --steps 20 --cfg-scale 4 \
  --seed 42 \
  --audio \
  --output-path "$OUT/test_dev_2bit_audio_5s.mp4"

./.venv/bin/python -m mlx_video.generate \
  --prompt "$PROMPT" \
  --pipeline distilled \
  --model-repo "$DIS_8BIT" \
  --text-encoder-repo "$TEXT_ENCODER_REPO_Q" \
  --width 832 --height 512 \
  --fps 24 --num-frames 73 \
  --steps 8 --cfg-scale 4 \
  --seed 42 \
  --audio \
  --output-path "$OUT/test_distilled_8bit_audio_5s.mp4"

./.venv/bin/python -m mlx_video.generate \
  --prompt "$PROMPT" \
  --pipeline distilled \
  --model-repo "$DIS_4BIT" \
  --text-encoder-repo "$TEXT_ENCODER_REPO_Q" \
  --width 832 --height 512 \
  --fps 24 --num-frames 73 \
  --steps 8 --cfg-scale 4 \
  --seed 42 \
  --audio \
  --output-path "$OUT/test_distilled_4bit_audio_5s.mp4"

./.venv/bin/python -m mlx_video.generate \
  --prompt "$PROMPT" \
  --pipeline distilled \
  --model-repo "$DIS_2BIT" \
  --text-encoder-repo "$TEXT_ENCODER_REPO_Q" \
  --width 832 --height 512 \
  --fps 24 --num-frames 73 \
  --steps 8 --cfg-scale 4 \
  --seed 42 \
  --audio \
  --output-path "$OUT/test_distilled_2bit_audio_5s.mp4"

echo "==> Done"
