"""
detector.py
-----------
Thin wrapper around an Ultralytics YOLO model. Keeping this separate from
main.py means the tracking/analytics code never has to know which detector
is being used underneath.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils import ModelLoadError


@dataclass
class Detections:
    """Plain container for one frame's detections."""
    boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))   # x1,y1,x2,y2
    scores: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    class_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))

    def __len__(self):
        return len(self.boxes)


class YoloDetector:
    """Loads a YOLO model once and exposes a simple `detect(frame)` call."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.4,
                 classes: Optional[list] = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError(
                "The 'ultralytics' package is not installed. Run: pip install ultralytics"
            ) from exc

        try:
            self.model = YOLO(weights)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load YOLO weights '{weights}'. "
                "Check the model name/path and your internet connection "
                "(weights are auto-downloaded on first use)."
            ) from exc

        self.conf = conf
        self.classes = classes
        self.class_names = self.model.names  # {id: name}

    def detect(self, frame) -> Detections:
        """Runs detection on a single BGR frame. Never raises on a bad frame;
        returns an empty Detections object instead so the main loop can continue."""
        if frame is None or frame.size == 0:
            return Detections()

        try:
            result = self.model.predict(
                frame, conf=self.conf, classes=self.classes, verbose=False
            )[0]
        except Exception as exc:  # keep the app alive on an occasional bad frame
            print(f"[WARN] Detection failed on this frame: {exc}")
            return Detections()

        if len(result.boxes) == 0:
            return Detections()

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        return Detections(boxes, scores, class_ids)

    def class_name(self, class_id: int) -> str:
        return self.class_names.get(int(class_id), "object")
