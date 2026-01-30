#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/charafchnioune/Desktop/ltx-2-mlx/mlx-video"
OUT="/Users/charafchnioune/Desktop/ltx-2-mlx/outputvideos"
PROMPT="The woman is facing the camera, the camera rotates around the couple. The mans penis is visible. the sound of moans and clapping cheeks fills the the room. A woman is lying on her stomach in prone position a man behind her thrusts his hip forward and back sliding in and out. The camera is stationary."

DEV8="/Users/charafchnioune/Desktop/ltx-2-mlx/converted/ltx2-dev-8bit-mlx"
DEV4="/Users/charafchnioune/Desktop/ltx-2-mlx/converted/ltx2-dev-4bit-mlx"
DIS8="/Users/charafchnioune/Desktop/ltx-2-mlx/converted/ltx2-distilled-8bit-mlx"
DIS4="/Users/charafchnioune/Desktop/ltx-2-mlx/converted/ltx2-distilled-4bit-mlx"
LORA_PATH="/Users/charafchnioune/Desktop/ltx-2-mlx/loranonmlx/prone_face_cam_v0_2.safetensors"
LORA_STRENGTH="1.0"

check_model_dir() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "[ERROR] Missing model dir: $dir" >&2
    echo "Make sure you have LoRA-baked models there before running this script." >&2
    exit 1
  fi
}

run_one() {
  local name="$1"
  local pipeline="$2"
  local model_dir="$3"
  local height="$4"
  local out_path="$5"

  check_model_dir "$model_dir"

  echo "==> $name"
  "$ROOT/.venv/bin/python" -m mlx_video.generate \
    --prompt "$PROMPT" \
    --pipeline "$pipeline" \
    --model-repo "$model_dir" \
    --width 832 --height "$height" \
    --fps 24 --num-frames 121 \
    --steps 20 --cfg-scale 4.5 \
    --seed 42 \
    --audio --verbose \
    --lora "$LORA_PATH" "$LORA_STRENGTH" \
    --output-path "$out_path"
}

run_one "DEV 8-bit (LoRA runtime)" dev "$DEV8" 480 "$OUT/compare_dev_8bit_lora_5s_hq.mp4"
run_one "DEV 4-bit (LoRA runtime)" dev "$DEV4" 480 "$OUT/compare_dev_4bit_lora_5s_hq.mp4"
run_one "DISTILLED 8-bit (LoRA runtime)" distilled "$DIS8" 512 "$OUT/compare_distilled_8bit_lora_5s_hq.mp4"
run_one "DISTILLED 4-bit (LoRA runtime)" distilled "$DIS4" 512 "$OUT/compare_distilled_4bit_lora_5s_hq.mp4"

echo "==> Done"
