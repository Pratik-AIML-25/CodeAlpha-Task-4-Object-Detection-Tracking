"""
tracker/sort.py
----------------
SORT (Simple Online and Realtime Tracking), implemented from scratch.

Each tracked object gets a Kalman Filter with a constant-velocity model over
[center_x, center_y, area, aspect_ratio]. Every frame, existing tracks predict
their next position, and the Hungarian algorithm matches those predictions to
the new frame's detections using IoU as the cost.

Class-aware association
------------------------
A person and an overlapping car (or a dog and a nearby backpack) can have
high IoU. To stop a track from "jumping" between classes -- e.g. Person ID 3
suddenly being reassigned to a car -- detections and tracks are only ever
matched against others of the *same class*. Each class is matched
independently, then the results are merged.

Reference: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
"""

from typing import List, Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """IoU between two sets of [x1,y1,x2,y2] boxes -> (len(bb_test), len(bb_gt)) matrix."""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])

    union = area_test + area_gt - intersection
    return intersection / np.maximum(union, 1e-6)


def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] -> [cx,cy,area,aspect_ratio], as a column vector."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([cx, cy, s, r]).reshape((4, 1))


def z_to_bbox(z: np.ndarray) -> np.ndarray:
    """[cx,cy,area,aspect_ratio] -> [x1,y1,x2,y2]."""
    w = np.sqrt(max(z[2] * z[3], 0.0))
    h = z[2] / (w + 1e-6)
    x1, y1 = z[0] - w / 2.0, z[1] - h / 2.0
    x2, y2 = z[0] + w / 2.0, z[1] + h / 2.0
    return np.array([x1, y1, x2, y2]).reshape((1, 4))


class KalmanBoxTracker:
    """One tracked object's Kalman filter, class label, and lifecycle counters."""

    count = 0

    def __init__(self, bbox: np.ndarray, class_id: int, score: float):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = bbox_to_z(bbox)

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.class_id = class_id
        self.score = score

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox: np.ndarray, score: float):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = score
        self.kf.update(bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return z_to_bbox(self.kf.x[:4].reshape(-1))

    def get_state(self) -> np.ndarray:
        return z_to_bbox(self.kf.x[:4].reshape(-1))


def associate(detections: np.ndarray, trackers: np.ndarray, iou_threshold: float):
    """Hungarian-algorithm matching within one class. Returns (matches, unmatched_d, unmatched_t)
    where indices are local to the arrays passed in."""
    if len(trackers) == 0 or len(detections) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.arange(len(trackers)))

    iou_matrix = iou_batch(detections, trackers)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matches, unmatched_d, unmatched_t = [], [], []
    matched_rows, matched_cols = set(), set()

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append([r, c])
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_d = [i for i in range(len(detections)) if i not in matched_rows]
    unmatched_t = [i for i in range(len(trackers)) if i not in matched_cols]

    matches = np.array(matches) if matches else np.empty((0, 2), dtype=int)
    return matches, np.array(unmatched_d), np.array(unmatched_t)


class Sort:
    """
    Multi-object tracker. Call `update()` once per frame, even with zero
    detections, so track ages advance correctly.
    """

    def __init__(self, max_age: int = 20, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        """
        boxes: (N,4) [x1,y1,x2,y2]; scores: (N,); class_ids: (N,)
        Returns (M,7): [x1,y1,x2,y2,track_id,class_id,score] for currently visible tracks.
        """
        self.frame_count += 1

        # 1. Predict every existing track's new position.
        predicted = np.zeros((len(self.trackers), 4))
        invalid = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            predicted[i] = pos
            if np.any(np.isnan(pos)):
                invalid.append(i)
        for i in reversed(invalid):
            self.trackers.pop(i)
            predicted = np.delete(predicted, i, axis=0)

        # 2. Match detections to tracks, one class at a time (class-aware SORT).
        matched_pairs = []
        unmatched_dets = set(range(len(boxes)))
        unmatched_trks = set(range(len(self.trackers)))

        classes_present = set(class_ids.tolist()) | {t.class_id for t in self.trackers}
        for cls in classes_present:
            det_idx = [i for i in range(len(boxes)) if class_ids[i] == cls]
            trk_idx = [i for i, t in enumerate(self.trackers) if t.class_id == cls]
            if not det_idx or not trk_idx:
                continue

            matches, _, _ = associate(boxes[det_idx][:, :4], predicted[trk_idx], self.iou_threshold)
            for d_local, t_local in matches:
                d_global, t_global = det_idx[d_local], trk_idx[t_local]
                matched_pairs.append((d_global, t_global))
                unmatched_dets.discard(d_global)
                unmatched_trks.discard(t_global)

        # 3. Update matched tracks with their assigned detection.
        for d, t in matched_pairs:
            self.trackers[t].update(boxes[d, :4], float(scores[d]))

        # 4. Start new tracks for detections nothing matched.
        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(boxes[d, :4], int(class_ids[d]), float(scores[d])))

        # 5. Build output for confirmed, currently-updated tracks; prune dead ones.
        output = []
        for trk in reversed(self.trackers):
            state = trk.get_state()[0]
            confirmed = trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            if trk.time_since_update < 1 and confirmed:
                output.append(np.concatenate((state, [trk.id, trk.class_id, trk.score])).reshape(1, -1))
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)

        return np.concatenate(output) if output else np.empty((0, 7))
