from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def read_video(path: str, max_frames: int | None = None) -> Tuple[np.ndarray, float]:
    """Read a video into an ndarray [T, H, W, C] and return fps."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return np.stack(frames, axis=0), fps


def save_video(frames: np.ndarray, path: str, fps: float = 24.0) -> None:
    """Save RGB frames to mp4."""
    path = str(path)
    h, w = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for f in frames:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def mux_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """Mux audio into an mp4 with ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
