import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType
from mlx_video.models.ltx.ltx import LTXModel
from mlx_video.models.ltx.attention import Attention


def get_model_path(
    path_or_hf_repo: str,
    revision: Optional[str] = None,
) -> Path:
    """Get local path to model, downloading if necessary.

    Args:
        path_or_hf_repo: Local path or HuggingFace repo ID
        revision: Git revision for HF repo

    Returns:
        Path to model directory
    """
    model_path = Path(path_or_hf_repo)

    if model_path.exists():
        return model_path

    # Download from HuggingFace
    model_path = Path(
        snapshot_download(
            repo_id=path_or_hf_repo,
            revision=revision,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "config.json",
            ],
        )
    )

    return model_path


def load_safetensors(path: Path) -> Dict[str, mx.array]:
    """Load weights from safetensors file(s) using MLX.

    Args:
        path: Path to model directory or single safetensors file

    Returns:
        Dictionary of weights
    """
    weights = {}

    if path.is_file():
        # Single file - use mx.load directly (handles bfloat16)
        return mx.load(str(path))
    else:
        # Directory - load all safetensors files
        safetensor_files = list(path.glob("*.safetensors"))
        for sf_path in safetensor_files:
            file_weights = mx.load(str(sf_path))
            weights.update(file_weights)

    return weights


def load_transformer_weights(model_path: Path) -> Dict[str, mx.array]:
    """Load transformer weights from LTX-2 model.

    Args:
        model_path: Path to LTX-2 model directory

    Returns:
        Dictionary of transformer weights
    """
    # Try distilled model first, then dev
    weight_files = [
        model_path / "ltx-2-19b-distilled.safetensors",
        model_path / "ltx-2-19b-dev.safetensors",
    ]

    for weight_file in weight_files:
        if weight_file.exists():
            print(f"Loading transformer weights from {weight_file.name}...")
            return mx.load(str(weight_file))

    raise FileNotFoundError(f"No transformer weights found in {model_path}")


def load_vae_weights(model_path: Path) -> Dict[str, mx.array]:
    """Load VAE weights from LTX-2 model.

    Args:
        model_path: Path to LTX-2 model directory

    Returns:
        Dictionary of VAE weights
    """
    vae_path = model_path / "vae" / "diffusion_pytorch_model.safetensors"
    if vae_path.exists():
        print(f"Loading VAE weights from {vae_path}...")
        return mx.load(str(vae_path))

    raise FileNotFoundError(f"VAE weights not found at {vae_path}")


def load_audio_vae_weights(model_path: Path) -> Dict[str, mx.array]:
    """Load audio VAE weights from LTX-2 model.

    Args:
        model_path: Path to LTX-2 model directory

    Returns:
        Dictionary of audio VAE weights
    """
    # Try different possible paths for audio VAE weights
    audio_vae_paths = [
        model_path / "audio_vae" / "diffusion_pytorch_model.safetensors",
        model_path / "audio_vae.safetensors",
    ]

    # Also check in main model weights
    main_paths = [
        model_path / "ltx-2-19b-distilled.safetensors",
        model_path / "ltx-2-19b-dev.safetensors",
    ]

    for audio_path in audio_vae_paths:
        if audio_path.exists():
            print(f"Loading audio VAE weights from {audio_path}...")
            return mx.load(str(audio_path))

    # Check main model weights for audio_vae keys
    for main_path in main_paths:
        if main_path.exists():
            print(f"Loading audio VAE weights from {main_path.name}...")
            all_weights = mx.load(str(main_path))
            # Filter to only audio_vae keys
            audio_weights = {k: v for k, v in all_weights.items() if "audio_vae" in k}
            if audio_weights:
                return audio_weights

    raise FileNotFoundError(f"Audio VAE weights not found in {model_path}")


def load_vocoder_weights(model_path: Path) -> Dict[str, mx.array]:
    """Load vocoder weights from LTX-2 model.

    Args:
        model_path: Path to LTX-2 model directory

    Returns:
        Dictionary of vocoder weights
    """
    # Try different possible paths for vocoder weights
    vocoder_paths = [
        model_path / "vocoder" / "diffusion_pytorch_model.safetensors",
        model_path / "vocoder.safetensors",
    ]

    # Also check in main model weights
    main_paths = [
        model_path / "ltx-2-19b-distilled.safetensors",
        model_path / "ltx-2-19b-dev.safetensors",
    ]

    for vocoder_path in vocoder_paths:
        if vocoder_path.exists():
            print(f"Loading vocoder weights from {vocoder_path}...")
            return mx.load(str(vocoder_path))

    # Check main model weights for vocoder keys
    for main_path in main_paths:
        if main_path.exists():
            print(f"Loading vocoder weights from {main_path.name}...")
            all_weights = mx.load(str(main_path))
            # Filter to only vocoder keys
            vocoder_weights = {k: v for k, v in all_weights.items() if "vocoder" in k}
            if vocoder_weights:
                return vocoder_weights

    raise FileNotFoundError(f"Vocoder weights not found in {model_path}")


def sanitize_transformer_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize transformer weight names from PyTorch LTX-2 format to MLX format.

    Args:
        weights: Dictionary of weights with PyTorch naming

    Returns:
        Dictionary with MLX-compatible naming for transformer
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Skip non-transformer weights (VAE, vocoder, audio_vae, connectors)
        if not key.startswith("model.diffusion_model."):
            continue

        # Remove 'model.diffusion_model.' prefix
        new_key = key.replace("model.diffusion_model.", "")

        # Handle to_out.0 -> to_out (MLX doesn't use Sequential numbering)
        new_key = new_key.replace(".to_out.0.", ".to_out.")

        # Handle feed-forward net naming
        # PyTorch: ff.net.0.proj -> ff.net_0_proj (or similar)
        # MLX FeedForward: uses proj_in, proj_out
        new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
        new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
        new_key = new_key.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
        new_key = new_key.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")

        # Handle AdaLN naming - keep emb wrapper, just fix linear naming
        # PyTorch: adaln_single.emb.timestep_embedder.linear_1 -> adaln_single.emb.timestep_embedder.linear1
        new_key = new_key.replace(".linear_1.", ".linear1.")
        new_key = new_key.replace(".linear_2.", ".linear2.")

        # Handle caption projection (keep linear1/linear2 naming for compatibility)
        # These are already mapped correctly in the sanitization

        sanitized[new_key] = value

    return sanitized


def sanitize_vae_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize VAE weight names from PyTorch format to MLX format.

    Args:
        weights: Dictionary of weights with PyTorch naming

    Returns:
        Dictionary with MLX-compatible naming for VAE decoder
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Skip position_ids (not needed)
        if "position_ids" in key:
            continue

        # Only process VAE decoder weights (skip audio_vae, etc.)
        if not key.startswith("vae."):
            continue

        # Handle per-channel statistics key mapping
        # PyTorch: vae.per_channel_statistics.mean-of-means -> per_channel_statistics.mean
        # PyTorch: vae.per_channel_statistics.std-of-means -> per_channel_statistics.std
        # Be careful: mean-of-stds_over_std-of-means also ends with std-of-means
        if "vae.per_channel_statistics" in key:
            if key == "vae.per_channel_statistics.mean-of-means":
                new_key = "per_channel_statistics.mean"
            elif key == "vae.per_channel_statistics.std-of-means":
                new_key = "per_channel_statistics.std"
            else:
                # Skip other per_channel_statistics keys (channel, mean-of-stds, etc.)
                continue
        elif key.startswith("vae.decoder."):
            # Strip the vae.decoder. prefix for decoder weights
            new_key = key.replace("vae.decoder.", "")
        else:
            # Skip other vae.* keys that are not decoder weights
            continue

        # Handle Conv3d weight shape conversion
        # PyTorch: (out_channels, in_channels, D, H, W)
        # MLX: (out_channels, D, H, W, in_channels)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 5:
            # Transpose from (O, I, D, H, W) to (O, D, H, W, I)
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Handle Conv2d weight shape conversion
        # PyTorch: (out_channels, in_channels, H, W)
        # MLX: (out_channels, H, W, in_channels)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        # Handle Conv1d weight shape conversion for vocoder blocks inside audio_vae
        # PyTorch: (out_channels, in_channels, kernel)
        # MLX: (out_channels, kernel, in_channels)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 3:
            value = mx.transpose(value, (0, 2, 1))

        sanitized[new_key] = value

    return sanitized


def sanitize_vae_encoder_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize VAE encoder weight names from PyTorch format to MLX format.

    Args:
        weights: Dictionary of weights with PyTorch naming

    Returns:
        Dictionary with MLX-compatible naming for VAE encoder
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Skip position_ids (not needed)
        if "position_ids" in key:
            continue

        # Only process VAE encoder weights
        if not key.startswith("vae."):
            continue

        # Handle per-channel statistics key mapping
        if "vae.per_channel_statistics" in key:
            if key == "vae.per_channel_statistics.mean-of-means":
                new_key = "per_channel_statistics._mean_of_means"
            elif key == "vae.per_channel_statistics.std-of-means":
                new_key = "per_channel_statistics._std_of_means"
            else:
                # Skip other per_channel_statistics keys
                continue
        elif key.startswith("vae.encoder."):
            # Strip the vae.encoder. prefix for encoder weights
            new_key = key.replace("vae.encoder.", "")
        else:
            # Skip other vae.* keys that are not encoder weights
            continue

        # Handle Conv3d weight shape conversion
        # PyTorch: (out_channels, in_channels, D, H, W)
        # MLX: (out_channels, D, H, W, in_channels)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Handle Conv2d weight shape conversion
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value

    return sanitized


def sanitize_audio_vae_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize audio VAE weight names from PyTorch format to MLX format.

    Args:
        weights: Dictionary of weights with PyTorch naming

    Returns:
        Dictionary with MLX-compatible naming for audio VAE encoder/decoder
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        if key.startswith("audio_vae.decoder."):
            new_key = key.replace("audio_vae.decoder.", "decoder.")
        elif key.startswith("audio_vae.encoder."):
            new_key = key.replace("audio_vae.encoder.", "encoder.")
        elif key.startswith("decoder."):
            new_key = key
        elif key.startswith("encoder."):
            new_key = key
        elif key.startswith("audio_vae.per_channel_statistics."):
            # Map per-channel statistics
            if "mean-of-means" in key:
                new_key = "per_channel_statistics._mean_of_means"
            elif "std-of-means" in key:
                new_key = "per_channel_statistics._std_of_means"
            else:
                continue  # Skip other statistics keys
        elif key in {"latents_mean", "latents_std"}:
            # audio_vae/diffusion_pytorch_model.safetensors
            new_key = "per_channel_statistics._mean_of_means" if key.endswith("mean") else "per_channel_statistics._std_of_means"
        else:
            continue  # Skip non-decoder keys

        # Handle Conv2d weight shape conversion
        # PyTorch: (out_channels, in_channels, H, W)
        # MLX: (out_channels, H, W, in_channels)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value

    return sanitized


def sanitize_vocoder_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize vocoder weight names from PyTorch format to MLX format.

    Args:
        weights: Dictionary of weights with PyTorch naming

    Returns:
        Dictionary with MLX-compatible naming for vocoder
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Handle vocoder weights
        if key.startswith("vocoder."):
            new_key = key.replace("vocoder.", "")

            # Handle ModuleList indices -> dict keys
            # PyTorch: ups.0, ups.1, ... -> ups.0, ups.1, ...
            # PyTorch: resblocks.0, resblocks.1, ... -> resblocks.0, resblocks.1, ...

            # Handle Conv1d weight shape conversion
            # PyTorch: (out_channels, in_channels, kernel)
            # MLX: (out_channels, kernel, in_channels)
            if "weight" in new_key and value.ndim == 3:
                if "ups" in new_key:
                    # ConvTranspose1d: PyTorch (in_ch, out_ch, kernel) -> MLX (out_ch, kernel, in_ch)
                    value = mx.transpose(value, (1, 2, 0))
                else:
                    # Conv1d: PyTorch (out_ch, in_ch, kernel) -> MLX (out_ch, kernel, in_ch)
                    value = mx.transpose(value, (0, 2, 1))

            sanitized[new_key] = value

    return sanitized


def sanitize_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize weight names from PyTorch format to MLX format.

    Generic function that handles both transformer and VAE weights.

    Args:
        weights: Dictionary of weights with PyTorch naming

    Returns:
        Dictionary with MLX-compatible naming
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Skip position_ids (not needed)
        if "position_ids" in key:
            continue

        # Handle transformer weights
        if key.startswith("model.diffusion_model."):
            new_key = key.replace("model.diffusion_model.", "")
            new_key = new_key.replace(".to_out.0.", ".to_out.")
            new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
            new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
            new_key = new_key.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
            new_key = new_key.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
            new_key = new_key.replace(".linear_1.", ".linear1.")
            new_key = new_key.replace(".linear_2.", ".linear2.")

        # Handle Conv3d weight shape conversion
        # PyTorch: (out_channels, in_channels, D, H, W)
        # MLX: (out_channels, D, H, W, in_channels)
        if "conv" in key.lower() and "weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Handle Conv2d weight shape conversion
        # PyTorch: (out_channels, in_channels, H, W)
        # MLX: (out_channels, H, W, in_channels)
        if "conv" in key.lower() and "weight" in key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value

    return sanitized


def load_config(model_path: Path) -> Dict[str, Any]:
    """Load model configuration.

    Args:
        model_path: Path to model directory

    Returns:
        Configuration dictionary
    """
    config_path = model_path / "config.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)

    # Return default config
    return {}


def _default_ltx_config() -> Dict[str, Any]:
    return {
        "model_type": "ltx av model",
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "in_channels": 128,
        "out_channels": 128,
        "num_layers": 48,
        "cross_attention_dim": 4096,
        "caption_channels": 3840,
        "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "audio_in_channels": 128,
        "audio_out_channels": 128,
        "audio_cross_attention_dim": 2048,
        "audio_caption_channels": 3840,
        "positional_embedding_theta": 10000.0,
        "positional_embedding_max_pos": [20, 2048, 2048],
        "audio_positional_embedding_max_pos": [20],
        "use_middle_indices_grid": True,
        "rope_type": "split",
        "double_precision_rope": True,
        "timestep_scale_multiplier": 1000,
        "av_ca_timestep_scale_multiplier": 1000,
        "norm_eps": 1e-6,
    }


def create_model_from_config(config: Dict[str, Any]) -> LTXModel:
    """Create model instance from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        LTXModel instance
    """
    # Map config to LTXModelConfig
    rope_type = config.get("rope_type", "split")
    if isinstance(rope_type, str):
        rope_type = LTXRopeType.SPLIT if rope_type.lower() == "split" else LTXRopeType.INTERLEAVED
    model_config = LTXModelConfig(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=config.get("num_attention_heads", 32),
        attention_head_dim=config.get("attention_head_dim", 128),
        in_channels=config.get("in_channels", 128),
        out_channels=config.get("out_channels", 128),
        num_layers=config.get("num_layers", 48),
        cross_attention_dim=config.get("cross_attention_dim", 4096),
        caption_channels=config.get("caption_channels", 3840),
        audio_num_attention_heads=config.get("audio_num_attention_heads", 32),
        audio_attention_head_dim=config.get("audio_attention_head_dim", 64),
        audio_in_channels=config.get("audio_in_channels", 128),
        audio_out_channels=config.get("audio_out_channels", 128),
        audio_cross_attention_dim=config.get("audio_cross_attention_dim", 2048),
        audio_caption_channels=config.get("audio_caption_channels", 3840),
        positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
        positional_embedding_max_pos=config.get("positional_embedding_max_pos", [20, 2048, 2048]),
        audio_positional_embedding_max_pos=config.get("audio_positional_embedding_max_pos", [20]),
        use_middle_indices_grid=config.get("use_middle_indices_grid", True),
        rope_type=rope_type,
        double_precision_rope=config.get("double_precision_rope", True),
        timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
        av_ca_timestep_scale_multiplier=config.get("av_ca_timestep_scale_multiplier", 1000),
        norm_eps=config.get("norm_eps", 1e-6),
    )

    return LTXModel(model_config)


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    dtype: Optional[str] = None,
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    q_mode: str = "affine",
    quantize_scope: str = "attn1",
    report_layers: bool = False,
    pipeline: str = "dev",
    loras: Optional[list[tuple[str, float]]] = None,
) -> Path:
    """Convert HuggingFace model to MLX format.

    Args:
        hf_path: HuggingFace model path or repo ID
        mlx_path: Output path for MLX model
        dtype: Target dtype (float16, float32, bfloat16)
        quantize: Whether to quantize the model
        q_bits: Quantization bits
        q_group_size: Quantization group size
        quantize_scope: Quantization scope ("attn1" or "all")
        report_layers: Whether to write a layer report JSON
        loras: Optional list of (path, strength) LoRA weights to merge before quantization

    Returns:
        Path to converted model
    """
    print(f"Loading model from {hf_path}...")
    model_path = get_model_path(hf_path)

    # Load config
    config = load_config(model_path)
    if not config:
        config = _default_ltx_config()

    # Load transformer weights for the requested pipeline
    print("Loading transformer weights...")
    if pipeline not in {"dev", "distilled"}:
        raise ValueError(f"Unsupported pipeline: {pipeline}")
    weight_file = model_path / f"ltx-2-19b-{pipeline}.safetensors"
    if not weight_file.exists():
        raise FileNotFoundError(f"Missing weight file: {weight_file}")
    raw_weights = mx.load(str(weight_file))

    # Extract text connector weights for the text encoder (small, needed for prompt conditioning)
    connector_weights: Dict[str, mx.array] = {}
    for key, value in raw_weights.items():
        if key.startswith("text_embedding_projection."):
            connector_weights[key] = value
        elif key.startswith("model.diffusion_model.video_embeddings_connector."):
            connector_weights[key] = value
        elif key.startswith("model.diffusion_model.audio_embeddings_connector."):
            connector_weights[key] = value

    # Sanitize transformer weights only
    print("Sanitizing transformer weights...")
    weights = sanitize_transformer_weights(raw_weights)

    # Convert dtype if specified
    if dtype is not None:
        dtype_map = {
            "float16": mx.float16,
            "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }
        target_dtype = dtype_map.get(dtype, mx.float16)
        print(f"Converting to {dtype}...")
        weights = {
            k: v.astype(target_dtype) if v.dtype in [mx.float32, mx.float16, mx.bfloat16] else v
            for k, v in weights.items()
        }

    # Apply LoRA deltas if requested (merge before quantization)
    if loras:
        from mlx_video.lora import LoraSpec, apply_lora_to_weights
        lora_specs = [LoraSpec(Path(path), float(strength)) for path, strength in loras]
        weights = apply_lora_to_weights(weights, lora_specs, verbose=True)

    # Quantize if requested
    if quantize:
        supported = {
            "affine": {"bits": {2, 3, 4, 5, 6, 8}, "group_size": {32, 64, 128}},
            "mxfp4": {"bits": {4}, "group_size": {32}},
            "mxfp8": {"bits": {8}, "group_size": {32}},
            "nvfp4": {"bits": {4}, "group_size": {16}},
        }
        if q_mode not in supported:
            raise ValueError(f"Unsupported quantization mode: {q_mode}")
        allowed_bits = supported[q_mode]["bits"]
        allowed_groups = supported[q_mode]["group_size"]
        if q_bits not in allowed_bits:
            raise ValueError(f"q_bits={q_bits} not supported for mode={q_mode}. Allowed: {sorted(allowed_bits)}")
        if q_group_size not in allowed_groups:
            raise ValueError(
                f"q_group_size={q_group_size} not supported for mode={q_mode}. Allowed: {sorted(allowed_groups)}"
            )
        print(f"Quantizing model (group_size={q_group_size}, bits={q_bits}, mode={q_mode})...")
        model = create_model_from_config(config)
        model.load_weights(list(weights.items()), strict=False)
        def _attn1_only_predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            # Quantize only video attention projections
            if "transformer_blocks" not in path:
                return False
            if "audio_" in path or "audio_to_video" in path or "video_to_audio" in path:
                return False
            if ".attn1" not in path:
                return False
            return True

        def _core_predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if "transformer_blocks" not in path:
                return False
            if ".attn" in path or ".ff" in path:
                return True
            if "audio_attn" in path or "audio_ff" in path:
                return True
            if "audio_to_video_attn" in path or "video_to_audio_attn" in path:
                return True
            return False

        def _all_quantizable_predicate(path, module):
            return hasattr(module, "to_quantized")

        nn.quantize(
            model,
            group_size=q_group_size,
            bits=q_bits,
            mode=q_mode,
            class_predicate=(
                _all_quantizable_predicate
                if quantize_scope == "all"
                else (_core_predicate if quantize_scope == "core" else _attn1_only_predicate)
            ),
        )
        from mlx.utils import tree_flatten
        weights = tree_flatten(model.parameters(), destination={})
        quant_tensors = sum(1 for v in weights.values() if v.dtype == mx.uint32)
        scale_tensors = sum(1 for k in weights.keys() if ".scales" in k or k.endswith("scales"))
        bias_tensors = sum(1 for k in weights.keys() if ".biases" in k or ".zeros" in k)
        print(
            f"Quantized tensors: {quant_tensors}, scale tensors: {scale_tensors}, "
            f"bias/zero tensors: {bias_tensors}"
        )

    # Create output directory
    output_path = Path(mlx_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights (MLX safetensors). This preserves bfloat16 and quantized weights.
    out_name = f"ltx-2-19b-{pipeline}-mlx.safetensors"
    print(f"Saving weights to {output_path / out_name}...")
    mx.save_safetensors(str(output_path / out_name), weights)

    # Save text connector weights (if available)
    if connector_weights:
        connectors_dir = output_path / "connectors"
        connectors_dir.mkdir(parents=True, exist_ok=True)
        connectors_path = connectors_dir / "ltx_text_connectors.safetensors"
        mx.save_safetensors(str(connectors_path), connector_weights)

    # Save config
    config_out_path = output_path / "config.json"
    with open(config_out_path, "w") as f:
        json.dump(config, f, indent=2)

    if quantize:
        meta = {
            "group_size": q_group_size,
            "bits": q_bits,
            "mode": q_mode,
            "dtype": dtype or "float16",
            "pipeline": pipeline,
            "predicate": "scales" if quantize_scope == "all" else ("core" if quantize_scope == "core" else "attn1_only"),
            "quantize_scope": quantize_scope,
        }
        with open(output_path / "quantization.json", "w") as f:
            json.dump(meta, f, indent=2)

    if report_layers:
        report = {
            "pipeline": pipeline,
            "quantize": quantize,
            "quantize_scope": quantize_scope,
            "dtype": dtype or "float16",
            "layers": [],
        }
        dtype_totals = {}
        total_bytes = 0
        for name, tensor in weights.items():
            nbytes = int(tensor.nbytes)
            total_bytes += nbytes
            dtype_key = str(tensor.dtype)
            dtype_totals[dtype_key] = dtype_totals.get(dtype_key, 0) + nbytes
            report["layers"].append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": dtype_key,
                    "nbytes": nbytes,
                }
            )
        report["totals"] = {
            "bytes": total_bytes,
            "by_dtype": dtype_totals,
        }
        report_path = output_path / "layer_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote layer report to {report_path}")

    print(f"Model converted successfully to {output_path}")
    return output_path


def save_weights(path: Path, weights: Dict[str, mx.array]) -> None:
    """Save weights in safetensors format.

    Args:
        path: Output directory
        weights: Dictionary of weights
    """
    from safetensors.numpy import save_file
    import numpy as np

    # Convert to numpy for safetensors
    np_weights = {k: np.array(v) for k, v in weights.items()}

    # Save to file
    save_file(np_weights, path / "model.safetensors")


def load_model(
    path_or_hf_repo: str,
    lazy: bool = False,
) -> LTXModel:
    """Load LTX model from path or HuggingFace.

    Args:
        path_or_hf_repo: Path to model or HuggingFace repo ID
        lazy: Whether to use lazy loading

    Returns:
        Loaded LTXModel
    """
    model_path = get_model_path(path_or_hf_repo)

    # Load config
    config = load_config(model_path)

    # Create model
    model = create_model_from_config(config)

    # Load weights
    weights = load_safetensors(model_path)

    # Sanitize if needed
    weights = sanitize_weights(weights)

    # Load weights into model
    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert LTX-2 model to MLX format")
    parser.add_argument(
        "--hf-path",
        type=str,
        default="Lightricks/LTX-2",
        help="HuggingFace model path or repo ID",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="Output path for MLX model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="Target dtype",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the model",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        help="Quantization bits",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Quantization group size",
    )
    parser.add_argument(
        "--q-mode",
        type=str,
        default="affine",
        help="Quantization mode",
    )
    parser.add_argument(
        "--quantize-scope",
        type=str,
        choices=["attn1", "core", "all"],
        default="attn1",
        help="Quantization scope: attn1, core (attn+ff), or all quantizable layers",
    )
    parser.add_argument(
        "--report-layers",
        action="store_true",
        help="Write layer_report.json with per-layer dtype/size",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["dev", "distilled"],
        default="dev",
        help="Pipeline weights to convert",
    )
    parser.add_argument(
        "--lora",
        nargs=2,
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="LoRA weights to merge before quantization (can be used multiple times): --lora path 0.8",
    )

    args = parser.parse_args()

    convert(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        dtype=args.dtype,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        q_mode=args.q_mode,
        quantize_scope=args.quantize_scope,
        report_layers=args.report_layers,
        pipeline=args.pipeline,
        loras=args.lora,
    )
