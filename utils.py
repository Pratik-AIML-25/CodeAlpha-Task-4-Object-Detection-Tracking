"""
utils.py
--------
Small, shared helpers used across the project: opening video sources safely,
an FPS counter, and a deterministic color palette for drawing.
"""

import time
from collections import deque

import cv2
import numpy as np


class VideoSourceError(Exception):
    """Raised when a webcam or video file cannot be opened/read."""


class ModelLoadError(Exception):
    """Raised when the detection model fails to load."""


def open_video_source(source: str) -> cv2.VideoCapture:
    """
    Opens a webcam (if `source` is a digit, e.g. "0") or a video file.
    Raises VideoSourceError with a clear message on failure instead of
    letting OpenCV fail silently.
    """
    cam_index = int(source) if str(source).isdigit() else None
    cap = cv2.VideoCapture(cam_index if cam_index is not None else source)

    if not cap.isOpened():
        if cam_index is not None:
            raise VideoSourceError(
                f"Could not open webcam index {cam_index}. "
                "Check that a camera is connected and not in use by another app."
            )
        raise VideoSourceError(
            f"Could not open video file: '{source}'. Check the path and that "
            "the file exists / is a supported format."
        )
    return cap


def make_video_writer(path: str, fps: float, size: tuple) -> cv2.VideoWriter:
    """Creates an mp4 writer, raising a clear error if it fails to open."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps if fps > 0 else 25.0, size)
    if not writer.isOpened():
        raise VideoSourceError(f"Could not create output video file at '{path}'.")
    return writer


class FPSCounter:
    """Rolling-average FPS counter (smoother than a single frame-to-frame delta)."""

    def __init__(self, window: int = 20):
        self._times = deque(maxlen=window)
        self._last = time.time()

    def tick(self) -> float:
        """Call once per frame. Returns the current smoothed FPS."""
        now = time.time()
        dt = now - self._last
        self._last = now
        if dt > 0:
            self._times.append(1.0 / dt)
        return sum(self._times) / len(self._times) if self._times else 0.0


def get_color(track_id: int) -> tuple:
    """Deterministic BGR color per track ID, so a box keeps its color over time."""
    rng = np.random.default_rng(int(track_id) * 7 + 3)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))
