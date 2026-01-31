"""Audio processing utilities for audio VAE."""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

try:  # optional dependency
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None


@dataclass
class AudioLatentShape:
    """Shape descriptor for audio latent representations."""

    batch: int
    channels: int
    frames: int
    mel_bins: int


class PerChannelStatistics(nn.Module):
    """
    Per-channel statistics for normalizing and denormalizing the latent representation.
    This statistics is computed over the entire dataset and stored in model's checkpoint.
    """

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        # Initialize buffers - will be loaded from weights
        # Using underscores for MLX compatibility with weight loading
        self._std_of_means = mx.ones((latent_channels,))
        self._mean_of_means = mx.zeros((latent_channels,))

    def un_normalize(self, x: mx.array) -> mx.array:
        """Denormalize latent representation."""
        # Broadcast statistics to match x shape
        # x shape: (B, C, ...) or (B, ..., C)
        std = self._std_of_means.astype(x.dtype)
        mean = self._mean_of_means.astype(x.dtype)
        return (x * std) + mean

    def normalize(self, x: mx.array) -> mx.array:
        """Normalize latent representation."""
        std = self._std_of_means.astype(x.dtype)
        mean = self._mean_of_means.astype(x.dtype)
        return (x - mean) / std


class AudioPatchifier:
    """
    Audio patchifier for converting between audio latents and patches.
    Combines channels and mel_bins dimensions for per-channel statistics.
    """

    def __init__(
        self,
        patch_size: int = 1,
        audio_latent_downsample_factor: int = 4,
        sample_rate: int = 16000,
        hop_length: int = 160,
        is_causal: bool = True,
    ):
        self.patch_size = patch_size
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.is_causal = is_causal

    def patchify(self, x: mx.array) -> mx.array:
        """Convert audio latents to patches.

        Input shape: (B, T, F, C) in MLX format (channels last)
        Output shape: (B, T, C*F) - flattened for per-channel statistics

        The output order is (c f) to match PyTorch's "b c t f -> b t (c f)".
        """
        # x shape: (B, T, F, C) e.g., (1, 68, 16, 8)
        b, t, f, c = x.shape
        # Transpose to (B, T, C, F) for correct (c f) ordering
        x = mx.transpose(x, (0, 1, 3, 2))
        # Reshape to (B, T, C*F) e.g., (1, 68, 128)
        return x.reshape(b, t, c * f)

    def unpatchify(self, x: mx.array, latent_shape: AudioLatentShape) -> mx.array:
        """Convert patches back to audio latents.

        Input shape: (B, T, C*F)
        Output shape: (B, T, F, C) in MLX format

        Reverses patchify's "b t (c f) -> b c t f" then transposes to MLX format.
        """
        # x shape: (B, T, C*F) e.g., (1, 68, 128)
        b, t, cf = x.shape
        c = latent_shape.channels
        f = latent_shape.mel_bins
        # Reshape to (B, T, C, F)
        x = x.reshape(b, t, c, f)
        # Transpose to MLX format (B, T, F, C)
        return mx.transpose(x, (0, 1, 3, 2))


class AudioProcessor:
    """Converts waveform audio into log-mel spectrograms (MLX).

    This mirrors the PyTorch AudioProcessor used in LTX-2 trainer.
    Uses soundfile to read waveforms. Falls back to a clear error
    if soundfile is unavailable.
    """

    def __init__(self, sample_rate: int, mel_bins: int, mel_hop_length: int, n_fft: int) -> None:
        self.sample_rate = int(sample_rate)
        self.mel_bins = int(mel_bins)
        self.mel_hop_length = int(mel_hop_length)
        self.n_fft = int(n_fft)

    def _load_waveform(self, path: str) -> tuple[np.ndarray, int]:
        if sf is None:
            raise RuntimeError("soundfile is required for audio preprocessing. Please install soundfile.")
        wav, sr = sf.read(path, always_2d=True)
        # sf returns (frames, channels); transpose to (channels, frames)
        return wav.T.astype(np.float32), int(sr)

    def _resample(self, waveform: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate == self.sample_rate:
            return waveform
        # naive resample using linear interpolation (no scipy dependency)
        ratio = self.sample_rate / float(source_rate)
        num = int(round(waveform.shape[1] * ratio))
        xp = np.linspace(0, 1, waveform.shape[1])
        xq = np.linspace(0, 1, num)
        out = np.stack([np.interp(xq, xp, ch) for ch in waveform], axis=0)
        return out.astype(np.float32)

    def _stft(self, waveform: np.ndarray) -> np.ndarray:
        # waveform: (channels, samples)
        win = np.hanning(self.n_fft).astype(np.float32)
        hop = self.mel_hop_length
        n_fft = self.n_fft
        frames = 1 + max((waveform.shape[1] - n_fft) // hop, 0)
        if frames <= 0:
            return np.zeros((waveform.shape[0], n_fft // 2 + 1, 1), dtype=np.float32)
        stft = []
        for ch in waveform:
            spectra = []
            for i in range(frames):
                start = i * hop
                frame = ch[start : start + n_fft]
                if frame.shape[0] < n_fft:
                    frame = np.pad(frame, (0, n_fft - frame.shape[0]))
                frame = frame * win
                spec = np.fft.rfft(frame)
                spectra.append(spec)
            stft.append(np.stack(spectra, axis=1))
        return np.stack(stft, axis=0)

    def _mel_filter(self) -> np.ndarray:
        # Simple mel filterbank (triangular). Avoids librosa dependency.
        sr = self.sample_rate
        n_fft = self.n_fft
        n_mels = self.mel_bins
        f_min = 0.0
        f_max = sr / 2.0

        def hz_to_mel(hz: float) -> float:
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel: float) -> float:
            return 700.0 * (10 ** (mel / 2595.0) - 1.0)

        m_min = hz_to_mel(f_min)
        m_max = hz_to_mel(f_max)
        m_pts = np.linspace(m_min, m_max, n_mels + 2)
        f_pts = mel_to_hz(m_pts)
        bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)

        fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for i in range(n_mels):
            left, center, right = bins[i], bins[i + 1], bins[i + 2]
            if center == left:
                center += 1
            if right == center:
                right += 1
            for j in range(left, center):
                fb[i, j] = (j - left) / float(center - left)
            for j in range(center, right):
                fb[i, j] = (right - j) / float(right - center)
        return fb

    def waveform_to_mel(self, waveform: np.ndarray, waveform_sample_rate: int) -> np.ndarray:
        waveform = self._resample(waveform, waveform_sample_rate)
        stft = self._stft(waveform)
        mag = np.abs(stft)
        fb = self._mel_filter()
        mel = np.matmul(fb, mag)  # (mel, freq) x (ch, freq, time) -> (mel, ch, time)
        mel = np.transpose(mel, (1, 0, 2))  # (ch, mel, time)
        mel = np.log(np.clip(mel, 1e-5, None))
        # Return (batch=1, channels, time, mel)
        return mel[None, ...].astype(np.float32)

    def load_audio_mel(self, path: str) -> np.ndarray:
        waveform, sr = self._load_waveform(path)
        return self.waveform_to_mel(waveform, sr)
