import math
from collections.abc import Generator, Iterator
from fractions import Fraction
from io import BytesIO

import av
import numpy as np
import mlx.core as mx
import cv2
from PIL import Image
from tqdm import tqdm

from ltx_pipelines.utils.constants import DEFAULT_IMAGE_CRF


def _to_numpy(value):
    if isinstance(value, mx.array):
        return np.array(value)
    return value


def resize_aspect_ratio_preserving(image: np.ndarray | mx.array, long_side: int) -> np.ndarray:
    """
    Resize image preserving aspect ratio (filling target long side).
    Preserves the input dimensions order.
    Args:
        image: Input image array with shape (F (optional), H, W, C)
        long_side: Target long side size.
    Returns:
        Array with shape (F (optional), H, W, C) or (H, W, C)
    """
    image_np = _to_numpy(image)
    height, width = image_np.shape[-3:2]
    max_side = max(height, width)
    scale = long_side / float(max_side)
    target_height = int(height * scale)
    target_width = int(width * scale)
    resized = resize_and_center_crop(image_np, target_height, target_width)
    # resized: (1, C, F, H, W)
    resized_np = _to_numpy(resized)
    result = resized_np.transpose(0, 2, 3, 4, 1)[0]  # (F, H, W, C)
    return result[0] if result.shape[0] == 1 else result


def resize_and_center_crop(tensor: np.ndarray | mx.array, height: int, width: int) -> mx.array:
    """
    Resize array preserving aspect ratio (filling target), then center crop to exact dimensions.
    Args:
        tensor: Input array with shape (H, W, C) or (F, H, W, C)
        height: Target height
        width: Target width
    Returns:
        MX array with shape (1, C, F, H, W)
    """
    arr = _to_numpy(tensor)
    if arr.ndim == 3:
        arr = arr[None, ...]  # F=1
    elif arr.ndim != 4:
        raise ValueError(f"Expected input with 3 or 4 dimensions; got shape {arr.shape}.")

    _, src_h, src_w, _ = arr.shape
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)

    out_frames = []
    for frame in arr:
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        crop_top = (new_h - height) // 2
        crop_left = (new_w - width) // 2
        frame_cropped = frame_resized[crop_top : crop_top + height, crop_left : crop_left + width]
        out_frames.append(frame_cropped)

    out = np.stack(out_frames, axis=0)  # (F, H, W, C)
    out = out.transpose(3, 0, 1, 2)  # (C, F, H, W)
    out = out[None, ...]  # (1, C, F, H, W)
    return mx.array(out)


def normalize_latent(latent: mx.array, dtype: mx.Dtype) -> mx.array:
    return (latent / 127.5 - 1.0).astype(dtype)


def load_image_conditioning(
    image_path: str, height: int, width: int, dtype: mx.Dtype, device: str
) -> mx.array:
    image = decode_image(image_path=image_path)
    image = preprocess(image=image)
    image = mx.array(image, dtype=mx.float32)
    image = resize_and_center_crop(image, height, width)
    image = normalize_latent(image, dtype)
    return image


def load_video_conditioning(
    video_path: str, height: int, width: int, frame_cap: int, dtype: mx.Dtype, device: str
) -> mx.array:
    frames = decode_video_from_file(path=video_path, frame_cap=frame_cap, device=device)
    result = None
    for f in frames:
        frame = resize_and_center_crop(f.astype(mx.float32), height, width)
        frame = normalize_latent(frame, dtype)
        result = frame if result is None else mx.concatenate([result, frame], axis=2)
    if result is None:
        raise ValueError(f"No frames decoded from {video_path}")
    return result


def decode_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    np_array = np.array(image)[..., :3]
    return np_array


def _write_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, samples: mx.array, audio_sample_rate: int
) -> None:
    samples_np = _to_numpy(samples)
    if samples_np.ndim == 1:
        samples_np = samples_np[:, None]

    if samples_np.shape[1] != 2 and samples_np.shape[0] == 2:
        samples_np = samples_np.T

    if samples_np.shape[1] != 2:
        raise ValueError(f"Expected samples with 2 channels; got shape {samples_np.shape}.")

    if samples_np.dtype != np.int16:
        samples_np = np.clip(samples_np, -1.0, 1.0)
        samples_np = (samples_np * 32767.0).astype(np.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples_np.reshape(1, -1),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate

    _resample_audio(container, audio_stream, frame_in)


def _prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _resample_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame
) -> None:
    cc = audio_stream.codec_context

    target_format = cc.format or "fltp"
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    for packet in audio_stream.encode():
        container.mux(packet)


def encode_video(
    video: mx.array | np.ndarray | Iterator[mx.array],
    fps: int,
    audio: mx.array | np.ndarray | None,
    audio_sample_rate: int | None,
    output_path: str,
    video_chunks_number: int,
) -> None:
    if isinstance(video, (mx.array, np.ndarray)):
        video = iter([video])

    first_chunk = next(video)
    first_chunk_np = _to_numpy(first_chunk)
    _, height, width, _ = first_chunk_np.shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    audio_stream = None
    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided")
        audio_stream = _prepare_audio_stream(container, audio_sample_rate)

    def all_tiles(
        first_chunk_local: np.ndarray, tiles_generator: Generator[tuple[np.ndarray, int], None, None]
    ) -> Generator[tuple[np.ndarray, int], None, None]:
        yield first_chunk_local
        yield from tiles_generator

    for video_chunk in tqdm(all_tiles(first_chunk_np, video), total=video_chunks_number):
        chunk_np = _to_numpy(video_chunk)
        if chunk_np.dtype != np.uint8:
            chunk_np = np.clip(chunk_np, 0, 255).astype(np.uint8)
        for frame_array in chunk_np:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    if audio_stream is not None and audio is not None:
        _write_audio(container, audio_stream, audio, audio_sample_rate)

    container.close()


def decode_audio_from_file(path: str, device: str) -> mx.array | None:
    container = av.open(path)
    try:
        audio = []
        audio_stream = next(s for s in container.streams if s.type == "audio")
        for frame in container.decode(audio_stream):
            audio.append(frame.to_ndarray().astype(np.float32))
        container.close()
        if not audio:
            return None
        audio_np = np.concatenate(audio, axis=0)
    except StopIteration:
        audio_np = None
    finally:
        container.close()

    if audio_np is None:
        return None
    return mx.array(audio_np)


def decode_video_from_file(path: str, frame_cap: int, device: str) -> Generator[mx.array, None, None]:
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        for frame in container.decode(video_stream):
            tensor = frame.to_rgb().to_ndarray().astype(np.uint8)
            tensor = tensor[None, ...]  # (1, H, W, C)
            yield mx.array(tensor)
            frame_cap = frame_cap - 1
            if frame_cap == 0:
                break
    finally:
        container.close()


def encode_single_frame(output_file: str, image_array: np.ndarray, crf: float) -> None:
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
        height = image_array.shape[0] // 2 * 2
        width = image_array.shape[1] // 2 * 2
        image_array = image_array[:height, :width]
        stream.height = height
        stream.width = width
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(format="yuv420p")
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def decode_single_frame(video_file: str) -> np.ndarray:
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def preprocess(image: np.ndarray, crf: float = DEFAULT_IMAGE_CRF) -> np.ndarray:
    if crf == 0:
        return image

    with BytesIO() as output_file:
        encode_single_frame(output_file, image, crf)
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    return image_array
