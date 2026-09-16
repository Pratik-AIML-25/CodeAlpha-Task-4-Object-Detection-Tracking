"""
tracker/deepsort_tracker.py
----------------------------
Thin adapter around the `deep-sort-realtime` package so that main.py can
treat SORT and DeepSORT interchangeably (same `update(boxes, scores,
class_ids, frame) -> Nx7 array` interface as tracker.sort.Sort).

DeepSORT adds a CNN appearance embedding for each detected crop, so
association uses both motion (IoU/Mahalanobis distance) and appearance
similarity. That makes it more robust across occlusions and crossing
objects than plain IoU matching, at extra compute cost per frame.

This is optional: if `deep-sort-realtime` isn't installed, importing this
module raises DeepSortUnavailableError, which main.py catches to give a
clear message instead of a crash.
"""

import numpy as np


class DeepSortUnavailableError(Exception):
    """Raised when the deep-sort-realtime package is not installed."""


class DeepSortTracker:
    def __init__(self, max_age: int = 20):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as exc:
            raise DeepSortUnavailableError(
                "DeepSORT was requested but 'deep-sort-realtime' is not installed. "
                "Run: pip install deep-sort-realtime  (or use --tracker sort)"
            ) from exc

        self._tracker = DeepSort(max_age=max_age)

    def update(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, frame=None) -> np.ndarray:
        detections = [
            ([b[0], b[1], b[2] - b[0], b[3] - b[1]], float(s), str(int(c)))
            for b, s, c in zip(boxes, scores, class_ids)
        ]
        tracks = self._tracker.update_tracks(detections, frame=frame)

        output = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = t.to_ltrb()
            cls_id = int(t.get_det_class()) if t.get_det_class() is not None else -1
            conf = t.get_det_conf()
            score = float(conf) if conf is not None else 0.0
            output.append([x1, y1, x2, y2, int(t.track_id) if str(t.track_id).isdigit() else hash(t.track_id) % 100000,
                            cls_id, score])

        return np.array(output) if output else np.empty((0, 7))
