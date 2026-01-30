#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
OUT="$REPO_ROOT/converted"
HF_REPO="${HF_REPO:-Lightricks/LTX-2}"

echo "==> Resolving HF snapshot for $HF_REPO"
MODEL_PATH="$("$ROOT"/.venv/bin/python - <<PY
from mlx_video.utils import get_model_path
print(get_model_path("$HF_REPO"))
PY
)"

mkdir -p "$OUT"
rm -rf "$OUT/ltx2-dev-8bit-mlx" "$OUT/ltx2-dev-4bit-mlx" \
  "$OUT/ltx2-distilled-8bit-mlx" "$OUT/ltx2-distilled-4bit-mlx"

copy_snapshot() {
  local dst="$1"
  mkdir -p "$dst"
  rsync -a "$MODEL_PATH/" "$dst/"
}

echo "==> Copying snapshot assets"
copy_snapshot "$OUT/ltx2-dev-8bit-mlx"
copy_snapshot "$OUT/ltx2-dev-4bit-mlx"
copy_snapshot "$OUT/ltx2-distilled-8bit-mlx"
copy_snapshot "$OUT/ltx2-distilled-4bit-mlx"

echo "==> Fixing symlinks to HF blobs"
"$ROOT"/.venv/bin/python - <<PY
import os
from pathlib import Path

snap = Path("$MODEL_PATH")
converted = Path("$OUT")

def fix_symlinks(root: Path):
    for p in root.rglob("*"):
        if p.is_symlink():
            target = os.readlink(p)
            rel = p.relative_to(root)
            orig_link_dir = (snap / rel).parent
            abs_target = (orig_link_dir / target).resolve()
            p.unlink()
            p.symlink_to(abs_target)

for name in ["ltx2-dev-8bit-mlx", "ltx2-dev-4bit-mlx", "ltx2-distilled-8bit-mlx", "ltx2-distilled-4bit-mlx"]:
    path = converted / name
    if path.exists():
        fix_symlinks(path)
PY

echo "==> Converting transformer weights (MLX quantized)"
"$ROOT"/.venv/bin/python -m mlx_video.convert \
  --hf-path "$HF_REPO" \
  --mlx-path "$OUT/ltx2-dev-8bit-mlx" \
  --dtype bfloat16 \
  --quantize --q-bits 8 --q-group-size 64 --q-mode affine --quantize-scope core --report-layers \
  --pipeline dev

"$ROOT"/.venv/bin/python -m mlx_video.convert \
  --hf-path "$HF_REPO" \
  --mlx-path "$OUT/ltx2-dev-4bit-mlx" \
  --dtype bfloat16 \
  --quantize --q-bits 4 --q-group-size 64 --q-mode affine --quantize-scope core --report-layers \
  --pipeline dev

"$ROOT"/.venv/bin/python -m mlx_video.convert \
  --hf-path "$HF_REPO" \
  --mlx-path "$OUT/ltx2-distilled-8bit-mlx" \
  --dtype bfloat16 \
  --quantize --q-bits 8 --q-group-size 64 --q-mode affine --quantize-scope core --report-layers \
  --pipeline distilled

"$ROOT"/.venv/bin/python -m mlx_video.convert \
  --hf-path "$HF_REPO" \
  --mlx-path "$OUT/ltx2-distilled-4bit-mlx" \
  --dtype bfloat16 \
  --quantize --q-bits 4 --q-group-size 64 --q-mode affine --quantize-scope core --report-layers \
  --pipeline distilled

echo "==> Done"
