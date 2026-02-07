from typing import List, Optional, Tuple
import os

import mlx.core as mx
import mlx.nn as nn
from pathlib import Path    
from mlx_video.models.ltx.config import (
    LTXModelConfig,
    LTXModelType,
    LTXRopeType,
    TransformerConfig,
)
from mlx_video.models.ltx.adaln import AdaLayerNormSingle
from mlx_video.models.ltx.rope import precompute_freqs_cis
from mlx_video.models.ltx.text_projection import PixArtAlphaTextProjection
from mlx_video.models.ltx.transformer import (
    BasicAVTransformerBlock,
    Modality,
    TransformerArgs,
)
from mlx_video.utils import to_denoised


def _debug_enabled() -> bool:
    return os.environ.get("LTX_DEBUG") == "1"


def _debug_log(message: str) -> None:
    if _debug_enabled():
        print(f"[debug][ltx] {message}")


class TransformerArgsPreprocessor:

    def __init__(
        self,
        patchify_proj: nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        inner_dim: int,
        max_pos: List[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        double_precision_rope: bool = False,
    ):
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope

    def _prepare_timestep(
        self,
        timestep: mx.array,
        batch_size: int,
        hidden_dtype: mx.Dtype = None,
    ) -> Tuple[mx.array, mx.array]:

        timestep = timestep * self.timestep_scale_multiplier
        timestep_emb, embedded_timestep = self.adaln(timestep.reshape(-1), hidden_dtype=hidden_dtype)

        # Reshape to (batch, tokens, dim)
        timestep_emb = mx.reshape(timestep_emb, (batch_size, -1, timestep_emb.shape[-1]))
        embedded_timestep = mx.reshape(embedded_timestep, (batch_size, -1, embedded_timestep.shape[-1]))

        return timestep_emb, embedded_timestep

    def _prepare_context(
        self,
        context: mx.array,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        batch_size = x.shape[0]

        # Context is already processed through embeddings connector in text encoder
        # Here we just apply the caption projection
        context = self.caption_projection(context)
        context = mx.reshape(context, (batch_size, -1, x.shape[-1]))
        return context, attention_mask

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[mx.array],
        x_dtype: mx.Dtype,
    ) -> Optional[mx.array]:
        if attention_mask is None:
            return None

        # Check if already float
        if attention_mask.dtype in [mx.float16, mx.float32, mx.bfloat16]:
            return attention_mask

        # Convert boolean/int mask to float mask
        # 0 -> -inf (masked), 1 -> 0 (not masked)
        mask = (attention_mask.astype(x_dtype) - 1) * 1e9
        mask = mx.reshape(mask, (attention_mask.shape[0], 1, -1, attention_mask.shape[-1]))
        return mask

    def _prepare_positional_embeddings(
        self,
        positions: mx.array,
        inner_dim: int,
        max_pos: List[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
    ) -> Tuple[mx.array, mx.array]:
        pe = precompute_freqs_cis(
            positions,
            dim=inner_dim,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            double_precision=self.double_precision_rope,
        )
        return pe

    def prepare(self, modality: Modality) -> TransformerArgs:
        x = self.patchify_proj(modality.latent)
        timestep, embedded_timestep = self._prepare_timestep(modality.timesteps, x.shape[0], hidden_dtype=x.dtype)
        context, attention_mask = self._prepare_context(modality.context, x, modality.context_mask)
        attention_mask = self._prepare_attention_mask(attention_mask, modality.latent.dtype)

        # Use precomputed positional embeddings if provided (avoids expensive RoPE recomputation)
        if modality.positional_embeddings is not None:
            pe = modality.positional_embeddings
        else:
            pe = self._prepare_positional_embeddings(
                positions=modality.positions,
                inner_dim=self.inner_dim,
                max_pos=self.max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                num_attention_heads=self.num_attention_heads,
            )

        return TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
        )


class MultiModalTransformerArgsPreprocessor:

    def __init__(
        self,
        patchify_proj: nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: List[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        av_ca_timestep_scale_multiplier: int,
        double_precision_rope: bool = False,
    ):
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def prepare(self, modality: Modality) -> TransformerArgs:
        from dataclasses import replace

        transformer_args = self.simple_preprocessor.prepare(modality)

        # Prepare cross-modal positional embeddings
        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
        )

        # Prepare cross-attention timestep embeddings
        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            timestep=modality.timesteps,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=transformer_args.x.shape[0],
            hidden_dtype=transformer_args.x.dtype,
        )

        return replace(
            transformer_args,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
        )

    def _prepare_cross_attention_timestep(
        self,
        timestep: mx.array,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: mx.Dtype = None,
    ) -> Tuple[mx.array, mx.array]:
        timestep = timestep * timestep_scale_multiplier

        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(timestep.reshape(-1), hidden_dtype=hidden_dtype)
        scale_shift_timestep = mx.reshape(scale_shift_timestep, (batch_size, -1, scale_shift_timestep.shape[-1]))

        gate_timestep, _ = self.cross_gate_adaln(timestep.reshape(-1) * av_ca_factor, hidden_dtype=hidden_dtype)
        gate_timestep = mx.reshape(gate_timestep, (batch_size, -1, gate_timestep.shape[-1]))

        return scale_shift_timestep, gate_timestep


class LTXModel(nn.Module):
 
    def __init__(self, config: LTXModelConfig):

        super().__init__()

        self.config = config
        self.model_type = config.model_type
        self.use_middle_indices_grid = config.use_middle_indices_grid
        self.rope_type = config.rope_type
        self.timestep_scale_multiplier = config.timestep_scale_multiplier
        self.positional_embedding_theta = config.positional_embedding_theta

        cross_pe_max_pos = None

        if config.model_type.is_video_enabled():
            self.positional_embedding_max_pos = config.positional_embedding_max_pos
            self.num_attention_heads = config.num_attention_heads
            self.inner_dim = config.inner_dim
            self._init_video(config)

        if config.model_type.is_audio_enabled():
            self.audio_positional_embedding_max_pos = config.audio_positional_embedding_max_pos
            self.audio_num_attention_heads = config.audio_num_attention_heads
            self.audio_inner_dim = config.audio_inner_dim
            self._init_audio(config)

        # Initialize cross-modal components
        if config.model_type.is_video_enabled() and config.model_type.is_audio_enabled():
            cross_pe_max_pos = max(
                config.positional_embedding_max_pos[0],
                config.audio_positional_embedding_max_pos[0],
            )
            self.av_ca_timestep_scale_multiplier = config.av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = config.audio_cross_attention_dim
            self._init_audio_video(config)

        self._init_preprocessors(config, cross_pe_max_pos)

        self._init_transformer_blocks(config)

    def _init_video(self, config: LTXModelConfig) -> None:
        self.patchify_proj = nn.Linear(config.in_channels, self.inner_dim, bias=True)
        self.adaln_single = AdaLayerNormSingle(self.inner_dim)
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=config.caption_channels,
            hidden_size=self.inner_dim,
        )

        self.scale_shift_table = mx.zeros((2, self.inner_dim))
        self.norm_out = nn.LayerNorm(self.inner_dim, eps=config.norm_eps, affine=False)
        self.proj_out = nn.Linear(self.inner_dim, config.out_channels)

    def _init_audio(self, config: LTXModelConfig) -> None:
        self.audio_patchify_proj = nn.Linear(config.audio_in_channels, self.audio_inner_dim, bias=True)
        self.audio_adaln_single = AdaLayerNormSingle(self.audio_inner_dim)

        # Audio caption projection: receives pre-processed embeddings from text encoder's audio_embeddings_connector
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=config.audio_caption_channels,
            hidden_size=self.audio_inner_dim,
        )

        # Output components
        self.audio_scale_shift_table = mx.zeros((2, self.audio_inner_dim))
        self.audio_norm_out = nn.LayerNorm(self.audio_inner_dim, eps=config.norm_eps, affine=False)
        self.audio_proj_out = nn.Linear(self.audio_inner_dim, config.audio_out_channels)

    def _init_audio_video(self, config: LTXModelConfig) -> None:
        num_scale_shift_values = 4

        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(self, config: LTXModelConfig, cross_pe_max_pos: Optional[int]) -> None:
        if config.model_type.is_video_enabled() and config.model_type.is_audio_enabled():
            # Multi-modal preprocessors
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=config.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=config.use_middle_indices_grid,
                audio_cross_attention_dim=config.audio_cross_attention_dim,
                timestep_scale_multiplier=config.timestep_scale_multiplier,
                positional_embedding_theta=config.positional_embedding_theta,
                rope_type=config.rope_type,
                av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
                double_precision_rope=config.double_precision_rope,
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=config.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=config.use_middle_indices_grid,
                audio_cross_attention_dim=config.audio_cross_attention_dim,
                timestep_scale_multiplier=config.timestep_scale_multiplier,
                positional_embedding_theta=config.positional_embedding_theta,
                rope_type=config.rope_type,
                av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
                double_precision_rope=config.double_precision_rope,
            )
        elif config.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                inner_dim=self.inner_dim,
                max_pos=config.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=config.use_middle_indices_grid,
                timestep_scale_multiplier=config.timestep_scale_multiplier,
                positional_embedding_theta=config.positional_embedding_theta,
                rope_type=config.rope_type,
                double_precision_rope=config.double_precision_rope,
            )
        elif config.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                inner_dim=self.audio_inner_dim,
                max_pos=config.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=config.use_middle_indices_grid,
                timestep_scale_multiplier=config.timestep_scale_multiplier,
                positional_embedding_theta=config.positional_embedding_theta,
                rope_type=config.rope_type,
                double_precision_rope=config.double_precision_rope,
            )

    def _init_transformer_blocks(self, config: LTXModelConfig) -> None:
        video_config = config.get_video_config()
        audio_config = config.get_audio_config()


        self.transformer_blocks = {
            idx: BasicAVTransformerBlock(
                idx=idx,
                video=video_config,
                audio=audio_config,
                rope_type=config.rope_type,
                norm_eps=config.norm_eps,
            )
            for idx in range(config.num_layers)
        }

    def _process_transformer_blocks(
        self,
        video: Optional[TransformerArgs],
        audio: Optional[TransformerArgs],
    ) -> Tuple[Optional[TransformerArgs], Optional[TransformerArgs]]:
        """Process through all transformer blocks."""
        for block in self.transformer_blocks.values():
            video, audio = block(video=video, audio=audio)
        return video, audio

    def _process_output(
        self,
        scale_shift_table: mx.array,
        norm_out: nn.LayerNorm,
        proj_out: nn.Linear,
        x: mx.array,
        embedded_timestep: mx.array,
    ) -> mx.array:
       
        # scale_shift_table: (2, dim) -> expand to (1, 1, 2, dim)
        # embedded_timestep: (B, 1, dim) -> expand to (B, 1, 1, dim)
        table_expanded = scale_shift_table[None, None, :, :]  # (1, 1, 2, dim)
        timestep_expanded = embedded_timestep[:, :, None, :]  # (B, 1, 1, dim)

        # Combine: (1, 1, 2, dim) + (B, 1, 1, dim) broadcasts to (B, 1, 2, dim)
        scale_shift_values = table_expanded + timestep_expanded

        # Extract shift and scale (first index is shift, second is scale)
        shift = scale_shift_values[:, :, 0, :]  # (B, 1, dim)
        scale = scale_shift_values[:, :, 1, :]  # (B, 1, dim)

        x = norm_out(x)
        x = x * (1 + scale) + shift  # Broadcasts (B, 1, dim) to (B, seq, dim)
        x = proj_out(x)

        return x

    def __call__(
        self,
        video: Optional[Modality] = None,
        audio: Optional[Modality] = None,
    ) -> Tuple[Optional[mx.array], Optional[mx.array]]:
       
        # Validate inputs
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        # Preprocess arguments
        video_args = self.video_args_preprocessor.prepare(video) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio) if audio is not None else None

        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
        )

        # Process outputs
        vx = (
            self._process_output(
                self.scale_shift_table,
                self.norm_out,
                self.proj_out,
                video_out.x,
                video_out.embedded_timestep,
            )
            if video_out is not None
            else None
        )

        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )

        return vx, ax

    def sanitize(self, weights: dict) -> dict:
        sanitized = {}

        for key, value in weights.items():
            new_key = key
            # Skip non-transformer weights (VAE, vocoder, audio_vae, connectors)
            if not key.startswith("model.diffusion_model.") or "audio_embeddings_connector" in key or "video_embeddings_connector" in key:
                continue

            # Remove 'model.diffusion_model.' prefix
            new_key = new_key.replace("model.diffusion_model.", "")

            new_key = new_key.replace(".to_out.0.", ".to_out.")

            new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
            new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
            new_key = new_key.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
            new_key = new_key.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")

            new_key = new_key.replace(".linear_1.", ".linear1.")
            new_key = new_key.replace(".linear_2.", ".linear2.")


            sanitized[new_key] = value

        return sanitized

    @classmethod
    def from_pretrained(
        cls,
        model_path: [Path, List[Path]],
        config: LTXModelConfig,
        strict: bool = True,
        weights_override: dict | None = None,
    ) -> None:
        from safetensors import safe_open

        model = cls(config)

        model_path_list = model_path if isinstance(model_path, list) else [model_path]
        def _sanitize_key(key: str) -> str | None:
            # Match `sanitize()` but only for the key mapping.
            if (
                (not key.startswith("model.diffusion_model."))
                or ("audio_embeddings_connector" in key)
                or ("video_embeddings_connector" in key)
            ):
                return None
            new_key = key.replace("model.diffusion_model.", "")
            new_key = new_key.replace(".to_out.0.", ".to_out.")
            new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
            new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
            new_key = new_key.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
            new_key = new_key.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
            new_key = new_key.replace(".linear_1.", ".linear1.")
            new_key = new_key.replace(".linear_2.", ".linear2.")
            return new_key

        def _scan_keys(paths: list[Path]) -> set[str]:
            keys: set[str] = set()
            for p in paths:
                try:
                    with safe_open(str(p), framework="numpy") as f:
                        keys.update(f.keys())
                except Exception:
                    # Best-effort fallback: if this isn't a safetensors file, we cannot cheaply scan keys.
                    pass
            return keys

        def _maybe_cast(key: str, value: mx.array, has_quant: bool) -> mx.array:
            if not hasattr(value, "dtype") or value.dtype != mx.float32:
                return value
            if has_quant and (key.endswith(".scales") or key.endswith(".biases")):
                return value
            return value.astype(mx.bfloat16)

        # Determine format/quantization without loading full tensors into a single dict.
        if weights_override is not None:
            # This path is used for unified weights or LoRA-merges, where tensors are already in memory.
            weights = dict(weights_override)
            is_pytorch = any(k.startswith("model.diffusion_model.") for k in weights)
            sanitized = model.sanitize(weights) if is_pytorch else weights
            has_quant = any(k.endswith(".scales") or k.endswith(".biases") for k in sanitized)
            scales_keys = {k for k in sanitized if k.endswith(".scales")}
        else:
            raw_keys = _scan_keys([Path(p) for p in model_path_list])
            is_pytorch = any(k.startswith("model.diffusion_model.") for k in raw_keys)
            if is_pytorch:
                mapped = [_sanitize_key(k) for k in raw_keys]
                mapped_keys = [k for k in mapped if k]
                has_quant = any(k.endswith(".scales") or k.endswith(".biases") for k in mapped_keys)
                scales_keys = {k for k in mapped_keys if k.endswith(".scales")}
            else:
                has_quant = any(k.endswith(".scales") or k.endswith(".biases") for k in raw_keys)
                scales_keys = {k for k in raw_keys if k.endswith(".scales")}

        if _debug_enabled():
            _debug_log(
                f"from_pretrained files={[str(p) for p in model_path_list]} "
                f"is_pytorch={is_pytorch} has_quant={has_quant} strict={strict} "
                f"scales_keys={len(scales_keys)}"
            )

        # If quantized weights are present, configure quantization before loading.
        if has_quant:
            # Default quant settings; override if quantization.json exists
            group_size = 64
            bits = 4
            mode = "affine"
            predicate = "scales"
            try:
                meta_path = Path(model_path_list[0]).parent / "quantization.json"
                if meta_path.exists():
                    import json
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    group_size = int(meta.get("group_size", group_size))
                    bits = int(meta.get("bits", bits))
                    mode = meta.get("mode", mode)
                    predicate = meta.get("predicate", predicate)
            except Exception:
                pass

            if _debug_enabled():
                _debug_log(
                    f"quantize: group_size={group_size} bits={bits} mode={mode} "
                    f"predicate={predicate}"
                )

            def _attn1_only_predicate(p, m):
                if not hasattr(m, "to_quantized"):
                    return False
                if "transformer_blocks" not in p:
                    return False
                if "audio_" in p or "audio_to_video" in p or "video_to_audio" in p:
                    return False
                if ".attn1" not in p:
                    return False
                return True

            def _core_predicate(p, m):
                if not hasattr(m, "to_quantized"):
                    return False
                if "transformer_blocks" not in p:
                    return False
                if ".attn" in p or ".ff" in p:
                    return True
                if "audio_attn" in p or "audio_ff" in p:
                    return True
                if "audio_to_video_attn" in p or "video_to_audio_attn" in p:
                    return True
                return False

            def _scales_predicate(p, m):
                return f"{p}.scales" in scales_keys

            # When loading already-quantized weights, rely on the saved .scales
            # tensors to decide which modules to quantize. This avoids mismatches
            # if quantization.json metadata is incomplete or uses a broader scope.
            if has_quant:
                pred = _scales_predicate
            elif predicate == "attn1_only":
                pred = _attn1_only_predicate
            elif predicate == "core":
                pred = _core_predicate
            else:
                pred = _scales_predicate

            nn.quantize(
                model,
                group_size=group_size,
                bits=bits,
                mode=mode,
                class_predicate=pred,
            )

            if _debug_enabled():
                q_modules = sum(1 for _, m in model.named_modules() if hasattr(m, "scales"))
                _debug_log(f"quantized modules={q_modules}")

        if weights_override is not None:
            # Cast in-memory overrides (used for LoRA merges) without duplicating scale/bias dtypes.
            sanitized = {
                k: _maybe_cast(k, v, has_quant)
                for k, v in sanitized.items()
            }
            model.load_weights(list(sanitized.items()), strict=strict)
        else:
            # Stream-load shards to reduce peak memory: do not aggregate into a huge dict first.
            for weight_file in model_path_list:
                shard = mx.load(str(weight_file))
                items = []
                if is_pytorch:
                    for k, v in shard.items():
                        kk = _sanitize_key(k)
                        if kk is None:
                            continue
                        items.append((kk, _maybe_cast(kk, v, has_quant)))
                else:
                    for k, v in shard.items():
                        items.append((k, _maybe_cast(k, v, has_quant)))
                if items:
                    model.load_weights(items, strict=strict)
                del shard
                mx.clear_cache()

        mx.eval(model.parameters())
        model.eval()
        return model


class X0Model(nn.Module):

    def __init__(self, velocity_model: LTXModel):
        
        super().__init__()
        self.velocity_model = velocity_model

    def __call__(
        self,
        video: Optional[Modality] = None,
        audio: Optional[Modality] = None,
    ) -> Tuple[Optional[mx.array], Optional[mx.array]]:
       
        vx, ax = self.velocity_model(video, audio)

        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None

        return denoised_video, denoised_audio
