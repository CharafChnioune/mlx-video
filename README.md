# mlx-video

MLX-Video is the best package for inference and finetuning of Image-Video-Audio generation models on your Mac using MLX.

## Installation

Install from source:

### Option 1: Install with pip (requires git):
```bash
pip install git+https://github.com/Blaizzy/mlx-video.git
```

### Option 2: Install with uv (ultra-fast package manager, optional):
```bash
uv pip install git+https://github.com/Blaizzy/mlx-video.git
```

Supported models:

### LTX-2
[LTX-2](https://huggingface.co/Lightricks/LTX-Video) is 19B parameter video generation model from Lightricks

## Features

- Text-to-video generation with the LTX-2 19B DiT model
- Two-stage distilled pipeline and single-stage dev pipeline
- Image, multi-image, keyframe (guide) conditioning
- Video conditioning (IC-LoRA style)
- Optional audio generation and streaming decode
- 2x spatial upscaling for images and videos (distilled)
- Optimized for Apple Silicon using MLX
- LoRA merge support for inference and LoRA-aware quant models


## Usage

### Text-to-Video Generation

```bash
uv run mlx_video.generate --prompt "Two dogs of the poodle breed wearing sunglasses, close up, cinematic, sunset" -n 100 --width 768
```

<img src="https://github.com/Blaizzy/mlx-video/raw/main/examples/poodles.gif" width="512" alt="Poodles demo">

With custom settings:

```bash
python -m mlx_video.generate \
    --prompt "Ocean waves crashing on a beach at sunset" \
    --height 768 \
    --width 768 \
    --num-frames 65 \
    --seed 123 \
    --output my_video.mp4
```

### Dev Pipeline (CFG, single-stage)

```bash
python -m mlx_video.generate \
    --prompt "A cinematic car drifting on a mountain road" \
    --pipeline dev \
    --steps 40 \
    --cfg-scale 4.5
```

### Image Conditioning (single or multi-image)

```bash
python -m mlx_video.generate \
    --prompt "A sunrise over a mountain valley" \
    --image first_frame.png 0 0.8 \
    --image later_frame.png 48 0.6
```

### Keyframe / Guide Mode

```bash
python -m mlx_video.generate \
    --prompt "A timelapse of a forest" \
    --pipeline keyframe \
    --conditioning-mode guide \
    --image keyframe.png 0 0.8
```

### Video Conditioning (IC-LoRA style)

```bash
python -m mlx_video.generate \
    --prompt "A dancer on stage" \
    --pipeline ic_lora \
    --video-conditioning reference.mp4 0 0.8
```

### Audio + Streaming

```bash
python -m mlx_video.generate \
    --prompt "Ocean waves with seagulls" \
    --audio \
    --stream
```

### Prompt Enhancement (optional)

```bash
# Default enhancer (uses the loaded text encoder)
python -m mlx_video.generate --prompt "A mountain lake at dawn" --enhance-prompt
```

### Auto Output Naming (optional)

```bash
python -m mlx_video.generate --prompt "A mountain lake at dawn" --auto-output-name
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | (required) | Text description of the video |
| `--height`, `-H` | 512 | Output height (divisible by 32 for dev, 64 for distilled) |
| `--width`, `-W` | 512 | Output width (divisible by 32 for dev, 64 for distilled) |
| `--num-frames`, `-n` | 33 | Number of frames (must be 1 + 8*k) |
| `--seed`, `-s` | 42 | Random seed for reproducibility |
| `--fps`, `--frame-rate` | 24 | Frames per second |
| `--output-path`, `--output`, `-o` | output.mp4 | Output video path |
| `--save-frames` | false | Save individual frames as images |
| `--model-repo` | Lightricks/LTX-2 | HuggingFace model repository |
| `--pipeline` | distilled | `distilled`, `dev`, `keyframe`, `ic_lora` |
| `--steps`, `--num-inference-steps` | 40 | Denoising steps |
| `--cfg-scale`, `--cfg-guidance-scale`, `--guidance-scale` | 4.0 | CFG guidance scale |
| `--image` | (none) | Image conditioning (repeatable) |
| `--video-conditioning` | (none) | Video conditioning (repeatable) |
| `--conditioning-mode` | replace | replace or guide |
| `--lora`, `--lora-path` | (none) | Merge LoRA weights (repeatable) |
| `--distilled-lora` | (none) | Stage-2 LoRA for distilled pipeline |
| `--audio` | false | Enable synchronized audio |
| `--enhance-prompt` | false | Enable prompt enhancement using the loaded text encoder |
| `--auto-output-name` | false | Auto-generate filename from prompt using Gemma |
| `--output-audio` | (none) | Save audio to custom path |
| `--stream` | false | Stream frames during decode |
| `--checkpoint-path`, `--checkpoint` | (none) | Optional explicit checkpoint path |
| `--gemma-root`, `--text-encoder-path` | (none) | Optional text encoder path |

### Trainer

There are two trainer entrypoints:

1) **MLX trainer (macOS, Apple Silicon)** — fully MLX-native for T2V and V2V (IC‑LoRA).
```bash
mlx_video.train --pipeline dev --training-mode lora --steps 1 --debug
```
You can also load LTX-2 style YAML configs:
```bash
mlx_video.train --config ltx_trainer/configs/ltx2_av_lora.yaml --debug
```
Validation sampling (optional):
```bash
mlx_video.train --steps 10 --validation-interval 5 --validation-prompts "A sunset over mountains" --debug
```
Strategies (text_to_video, video_to_video, ic_lora):
```bash
mlx_video.train --strategy video_to_video --data-root ./precomputed --debug
```
W&B and Hub upload (optional):
```bash
mlx_video.train --steps 5 --wandb-enabled --hub-push --hub-model-id username/my-ltx-lora
```

Precompute training data (latents + conditions):
```bash
mlx_video.precompute --input-dir ./videos --output-dir ./precomputed --prompts-file prompts.txt --debug
```
Auto‑caption if prompts are missing:
```bash
mlx_video.precompute --input-dir ./videos --output-dir ./precomputed --caption
```

Audio latents (optional):
```bash
mlx_video.precompute --input-dir ./videos --output-dir ./precomputed --with-audio --debug
```

Resolution buckets (optional):
```bash
mlx_video.precompute --input-dir ./videos --output-dir ./precomputed --resolution-buckets 832x480x73;768x768x65
```
```
Captioning (MLX backend) uses `mlx_vlm` by default (SmolVLM‑Instruct‑4bit).
To force Transformers (CPU) captioning:
```bash
mlx_video.precompute --input-dir ./videos --output-dir ./precomputed --caption --caption-backend transformers
```

2) **PyTorch/CUDA trainer** — not supported in this MLX-only fork. Use the MLX trainer above.

See `ltx_trainer/docs/mlx_limitations.md` for MLX redesign notes and CUDA‑only differences.

## How It Works

The pipeline uses a two-stage generation process:

1. **Stage 1**: Generate at half resolution (e.g., 384x384) with 8 denoising steps
2. **Upsample**: 2x spatial upsampling via LatentUpsampler
3. **Stage 2**: Refine at full resolution (e.g., 768x768) with 3 denoising steps
4. **Decode**: VAE decoder converts latents to RGB video

## Requirements

- macOS with Apple Silicon
- Python >= 3.11
- MLX >= 0.22.0

## Model Specifications

- **Transformer**: 48 layers, 32 attention heads, 128 dim per head
- **Latent channels**: 128
- **Text encoder**: Gemma 3 with 3840-dim output
- **RoPE**: Split mode with double precision

## Project Structure

```

## Quantized MLX Model Repos

You can pass these Hugging Face repos directly to `--model-repo` (they will auto-download if missing),
or use the short alias names shown below (handled automatically).

- `AITRADER/ltx2-dev-8bit-mlx` (alias: `ltx2-dev-8bit-mlx`)
- `AITRADER/ltx2-dev-4bit-mlx` (alias: `ltx2-dev-4bit-mlx`)
- `AITRADER/ltx2-distilled-8bit-mlx` (alias: `ltx2-distilled-8bit-mlx`)
- `AITRADER/ltx2-distilled-4bit-mlx` (alias: `ltx2-distilled-4bit-mlx`)

Example:
```bash
mlx_video.generate --pipeline dev --model-repo ltx2-dev-8bit-mlx --prompt "..." --output-path out.mp4
```

Notes:
- MLX affine quantization supports 2/3/4/5/6/8-bit with group_size 32/64/128. Lower bits are experimental and may reduce quality.
- To build local 2-bit models, set `ENABLE_2BIT=1` when running `scripts/build_converted_models.sh`.

## Supported vs Not Supported (MLX-only)

**Supported**
- MLX inference (dev + distilled), audio generation, streaming decode
- MLX LoRA runtime + quantized re-quantization
- MLX trainer (single-device) for T2V, V2V, IC-LoRA
- Quantized models via `--model-repo` aliases above

**Not supported (CUDA-only in PyTorch)**
- CUDA/Triton kernels, Accelerate/DeepSpeed distributed training
- bitsandbytes text-encoder 8-bit loading (use precomputed embeddings)

## PyTorch Pipeline Parity (MLX Wrappers)

The `ltx_pipelines` package in this repo now provides MLX wrappers that call
`mlx_video.generate` under the hood. You can run them as modules:

```bash
python -m ltx_pipelines.distilled --prompt "A sunrise over hills"
python -m ltx_pipelines.ic_lora --prompt "A dancer on stage" --video-conditioning ref.mp4 0 0.8
python -m ltx_pipelines.keyframe_interpolation --prompt "A forest" --image keyframe.png 0 0.8
python -m ltx_pipelines.ti2vid_one_stage --prompt "Ocean waves"
python -m ltx_pipelines.ti2vid_two_stages --prompt "Mountains" --image keyframe.png 0 0.8
```

These wrappers are MLX‑only (no CUDA) and share the same CLI flags as
`mlx_video.generate` (they inject `--pipeline` automatically).
mlx_video/
├── generate.py             # Video generation pipeline
├── convert.py              # Weight conversion (PyTorch -> MLX)
├── postprocess.py          # Video post-processing utilities
├── utils.py                # Helper functions
└── models/
    └── ltx/
        ├── ltx.py          # Main LTXModel (DiT transformer)
        ├── config.py       # Model configuration
        ├── transformer.py  # Transformer blocks
        ├── attention.py    # Multi-head attention with RoPE
        ├── text_encoder.py # Text encoder
        ├── upsampler.py    # 2x spatial upsampler
        └── video_vae/      # VAE encoder/decoder
```

## License

MIT
