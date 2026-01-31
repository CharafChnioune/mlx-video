from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import mlx.core as mx

from safetensors import safe_open

PRECOMPUTED_DIR_NAME = ".precomputed"


def _to_numpy(obj: Any) -> Any:
    if isinstance(obj, mx.array):
        return np.array(obj)
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_numpy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_numpy(v) for v in obj)
    return obj


def _load_pt(path: Path) -> Any:
    raise RuntimeError(f".pt files are not supported in the MLX-only trainer: {path}")


def _load_npz(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _load_safetensors(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with safe_open(str(path), framework="numpy") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def _load_any(path: Path) -> Dict[str, Any]:
    if path.suffix == ".pt":
        return _load_pt(path)
    if path.suffix == ".npz":
        return _load_npz(path)
    if path.suffix == ".safetensors":
        return _load_safetensors(path)
    raise ValueError(f"Unsupported file type: {path}")


@dataclass
class Batch:
    latents: Dict[str, Any]
    conditions: Dict[str, Any]
    audio_latents: Dict[str, Any] | None = None
    ref_latents: Dict[str, Any] | None = None


class DummyDataset:
    def __init__(
        self,
        width: int = 832,
        height: int = 480,
        num_frames: int = 33,
        fps: int = 24,
        dataset_length: int = 200,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 3840,
        prompt_sequence_length: int = 1024,
        with_audio: bool = False,
    ) -> None:
        if width % 32 != 0 or height % 32 != 0:
            raise ValueError("Width/height must be divisible by 32")
        if num_frames % 8 != 1:
            raise ValueError("num_frames must be 1 + 8*k")

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length
        self.with_audio = with_audio

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Batch:
        latents = {
            "latents": np.random.randn(
                self.latent_dim,
                self.num_latent_frames,
                self.latent_height,
                self.latent_width,
            ).astype(np.float32),
            "num_frames": np.array([self.num_latent_frames], dtype=np.int32),
            "height": np.array([self.latent_height], dtype=np.int32),
            "width": np.array([self.latent_width], dtype=np.int32),
            "fps": np.array([self.fps], dtype=np.float32),
        }

        conditions = {
            "video_prompt_embeds": np.random.randn(self.prompt_sequence_length, self.prompt_embed_dim).astype(np.float32),
            "audio_prompt_embeds": np.random.randn(self.prompt_sequence_length, self.prompt_embed_dim).astype(np.float32),
            "prompt_attention_mask": np.ones(self.prompt_sequence_length, dtype=bool),
        }

        audio_latents = None
        if self.with_audio:
            audio_latents = {
                "latents": np.random.randn(8, 69, 16).astype(np.float32),
                "num_time_steps": np.array([69], dtype=np.int32),
                "frequency_bins": np.array([16], dtype=np.int32),
            }

        return Batch(latents=latents, conditions=conditions, audio_latents=audio_latents)


class PrecomputedDataset:
    def __init__(self, data_root: str, data_sources: Dict[str, str] | List[str] | None = None) -> None:
        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths = self._setup_source_paths()
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        root = Path(data_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Data root does not exist: {root}")
        if (root / PRECOMPUTED_DIR_NAME).exists():
            root = root / PRECOMPUTED_DIR_NAME
        return root

    @staticmethod
    def _normalize_data_sources(data_sources: Dict[str, str] | List[str] | None) -> Dict[str, str]:
        if data_sources is None:
            return {"latents": "latents", "conditions": "conditions"}
        if isinstance(data_sources, list):
            return {name: name for name in data_sources}
        if isinstance(data_sources, dict):
            return data_sources.copy()
        raise TypeError(f"data_sources must be dict, list or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> Dict[str, Path]:
        source_paths: Dict[str, Path] = {}
        for dir_name in self.data_sources:
            path = self.data_root / dir_name
            if not path.exists():
                raise FileNotFoundError(f"Missing data source dir: {path}")
            source_paths[dir_name] = path
        return source_paths

    def _discover_samples(self) -> Dict[str, List[Path]]:
        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources.keys()))
        data_path = self.source_paths[data_key]
        data_files = list(data_path.glob("**/*"))
        data_files = [p for p in data_files if p.suffix in (".pt", ".npz", ".safetensors")]
        if not data_files:
            raise ValueError(f"No data files in {data_path}")

        sample_files: Dict[str, List[Path]] = {out_key: [] for out_key in self.data_sources.values()}
        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)
            if self._all_source_files_exist(data_file, rel_path):
                self._fill_sample_data_files(data_file, rel_path, sample_files)
        return sample_files

    def _all_source_files_exist(self, data_file: Path, rel_path: Path) -> bool:
        for dir_name in self.data_sources:
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            if not expected_path.exists():
                return False
        return True

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        source_path = self.source_paths[dir_name]
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}{data_file.suffix}"
        return source_path / rel_path

    def _fill_sample_data_files(self, data_file: Path, rel_path: Path, sample_files: Dict[str, List[Path]]) -> None:
        for dir_name, out_key in self.data_sources.items():
            expected = self._get_expected_file_path(dir_name, data_file, rel_path)
            sample_files[out_key].append(expected.relative_to(self.source_paths[dir_name]))

    def _validate_setup(self) -> None:
        if not self.sample_files:
            raise ValueError("No valid samples found")
        counts = {k: len(v) for k, v in self.sample_files.items()}
        if len(set(counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts: {counts}")

    def __len__(self) -> int:
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> Batch:
        result: Dict[str, Any] = {}
        for out_key, files in self.sample_files.items():
            source_dir = None
            # find which source directory maps to this output key
            for dir_name, mapped in self.data_sources.items():
                if mapped == out_key:
                    source_dir = self.source_paths[dir_name]
                    break
            if source_dir is None:
                raise RuntimeError(f"Missing source dir for {out_key}")
            path = source_dir / files[index]
            result[out_key] = _load_any(path)

        latents = result.get("latents")
        conditions = result.get("conditions") or result.get("text_conditions") or {}
        audio_latents = result.get("audio_latents")
        ref_latents = result.get("ref_latents")

        if latents is not None:
            latents = self._normalize_video_latents(latents)

        return Batch(latents=latents, conditions=conditions, audio_latents=audio_latents, ref_latents=ref_latents)

    @staticmethod
    def _normalize_video_latents(data: Dict[str, Any]) -> Dict[str, Any]:
        latents = data.get("latents")
        if not isinstance(latents, np.ndarray):
            latents = np.array(latents)
        # Legacy patchified format: [seq_len, C] -> [C, F, H, W]
        if latents.ndim == 2:
            num_frames = int(np.array(data["num_frames"]).reshape(-1)[0])
            height = int(np.array(data["height"]).reshape(-1)[0])
            width = int(np.array(data["width"]).reshape(-1)[0])
            latents = latents.reshape(num_frames, height, width, latents.shape[-1])
            latents = np.transpose(latents, (3, 0, 1, 2))
            data = data.copy()
            data["latents"] = latents
        return data

def collate_batches(batches: List[Batch]) -> Batch:
    def stack_dict(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = dicts[0].keys()
        out = {}
        for k in keys:
            vals = [d[k] for d in dicts]
            if isinstance(vals[0], np.ndarray):
                out[k] = np.stack(vals, axis=0)
            else:
                out[k] = np.array(vals)
        return out

    latents = stack_dict([b.latents for b in batches])
    conditions = stack_dict([b.conditions for b in batches])
    audio_latents = None
    if batches[0].audio_latents is not None:
        audio_latents = stack_dict([b.audio_latents for b in batches])
    ref_latents = None
    if batches[0].ref_latents is not None:
        ref_latents = stack_dict([b.ref_latents for b in batches])

    return Batch(latents=latents, conditions=conditions, audio_latents=audio_latents, ref_latents=ref_latents)


def iter_batches(dataset, batch_size: int, shuffle: bool = True, seed: int = 0):
    idxs = np.arange(len(dataset))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idxs)
    for i in range(0, len(dataset), batch_size):
        batch_idxs = idxs[i : i + batch_size]
        batch = [dataset[int(j)] for j in batch_idxs]
        yield collate_batches(batch)
