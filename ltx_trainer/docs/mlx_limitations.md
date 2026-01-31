# MLX Redesign Notes (CUDA‑only → MLX)

This repo is MLX‑only. CUDA/Triton features from the PyTorch trainer are redesigned
as MLX‑native equivalents or explicitly not supported.

## Training / Acceleration

- **Distributed training**: MLX trainer is single‑device only. Use gradient
  accumulation to simulate larger batch sizes. Multi‑device data‑parallel is not
  implemented in this repo.
- **CUDA/Triton kernels**: Not used. MLX kernels and vectorized ops are used
  instead.
- **Mixed precision**: MLX uses `float16`/`bfloat16` types. The trainer accepts
  `mixed_precision_mode` in YAML and logs a warning if unsupported behavior is
  requested.
- **Accelerate configs**: PyTorch Accelerate configs are not supported. Use
  MLX trainer CLI/YAML configs in `mlx_video.mlx_trainer`.

## Data / Precompute

- **Resolution buckets**: MLX precompute selects the nearest bucket and
  center‑crops after resize (matches PyTorch behavior).
- **Reference conditioning**: MLX `compute_reference.py` can update dataset
  JSON/CSV/JSONL with generated edge‑map references.

## Quantization

- **MLX quantization** uses `nn.quantize` (affine mode) with bits 2/3/4/5/6/8
  and group sizes 32/64/128. Lower bits are experimental.

