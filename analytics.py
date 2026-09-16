"""
analytics.py
------------
Everything that turns raw tracks into insight: motion trails, active/total
object counts, and IN/OUT counting across a virtual line. Kept separate from
drawing (visualization.py) and from the tracker itself.
"""

from collections import deque
from typing import Dict, List, Tuple


class TrajectoryManager:
    """Stores recent center points per track ID and drops history for tracks
    that haven't been seen in a while, so memory doesn't grow unbounded."""

    def __init__(self, trail_length: int = 30, ttl_frames: int = 30):
        self.trail_length = trail_length
        self.ttl_frames = ttl_frames
        self.history: Dict[int, deque] = {}
        self._last_seen: Dict[int, int] = {}

    def update(self, frame_idx: int, active_tracks: List[Tuple[int, float, float]]):
        """active_tracks: list of (track_id, center_x, center_y) for this frame."""
        for track_id, cx, cy in active_tracks:
            if track_id not in self.history:
                self.history[track_id] = deque(maxlen=self.trail_length)
            self.history[track_id].append((int(cx), int(cy)))
            self._last_seen[track_id] = frame_idx

        stale_ids = [tid for tid, last in self._last_seen.items()
                     if frame_idx - last > self.ttl_frames]
        for tid in stale_ids:
            self.history.pop(tid, None)
            self._last_seen.pop(tid, None)

    def get_trail(self, track_id: int) -> deque:
        return self.history.get(track_id, deque())


class ObjectCounter:
    """Tracks how many objects are currently active vs. how many unique
    tracking IDs have ever been seen (so re-detecting the same object across
    frames doesn't inflate the total)."""

    def __init__(self):
        self._seen_ids = set()

    def update(self, active_tracks: List[Tuple[int, str]]) -> dict:
        """active_tracks: list of (track_id, class_name) currently visible."""
        active_by_class: Dict[str, int] = {}
        for track_id, class_name in active_tracks:
            self._seen_ids.add(track_id)
            active_by_class[class_name] = active_by_class.get(class_name, 0) + 1

        return {
            "active_total": len(active_tracks),
            "active_by_class": active_by_class,
            "total_unique_tracked": len(self._seen_ids),
        }


class LineCounter:
    """
    Counts IN/OUT crossings of a horizontal virtual line, using a dead-zone
    (hysteresis band) around the line so that small frame-to-frame jitter
    near the line does not register as a crossing.

    Direction convention: a track moving from *above* the line to *below* it
    is counted OUT; below-to-above is counted IN. (Flip if your camera setup
    needs the opposite meaning.)

    Why hysteresis instead of a single line position
    --------------------------------------------------
    A plain "which side of the line is the center on" check flips every time
    a track's noisy center point wobbles across the exact line pixel, which
    can register several false crossings for one real event. Instead, two
    thresholds are used:

        upper_bound = line_y - margin      (clearly ABOVE the line)
        lower_bound = line_y + margin      (clearly BELOW the line)

    A track's "committed side" only updates once its center is clearly past
    one of these thresholds. While it's inside the dead zone between them,
    its committed side is left untouched -- so jitter that never leaves the
    dead zone, or that re-enters the dead zone without reaching the opposite
    threshold, never changes anything. A crossing is only counted when the
    committed side actually flips from one clear side to the other, which
    both requires meaningful movement across the line and guarantees each
    physical crossing is counted exactly once.
    """

    def __init__(self, line_y_fraction: float, margin_fraction: float = 0.02):
        """
        line_y_fraction: 0.0-1.0, fraction of frame height where the line sits.
        margin_fraction: half-width of the dead zone, as a fraction of frame
            height, on each side of the line (default 2% of frame height).
            Increase it for noisier tracks, decrease it for a more sensitive line.
        """
        self.line_y_fraction = line_y_fraction
        self.margin_fraction = margin_fraction
        self._committed_side: Dict[int, str] = {}  # track_id -> "above" | "below"
        self.count_in = 0
        self.count_out = 0

    def line_pixel_y(self, frame_height: int) -> int:
        return int(frame_height * self.line_y_fraction)

    def _margin_pixels(self, frame_height: int) -> float:
        return max(1.0, frame_height * self.margin_fraction)

    def update(self, frame_height: int, active_tracks: List[Tuple[int, float]]):
        """active_tracks: list of (track_id, center_y) for this frame."""
        line_y = self.line_pixel_y(frame_height)
        margin = self._margin_pixels(frame_height)
        upper_bound = line_y - margin
        lower_bound = line_y + margin

        for track_id, cy in active_tracks:
            if cy < upper_bound:
                candidate_side = "above"
            elif cy > lower_bound:
                candidate_side = "below"
            else:
                candidate_side = None  # inside the dead zone: ambiguous, ignore

            if candidate_side is None:
                continue

            prev_side = self._committed_side.get(track_id)

            if prev_side is None:
                # First time this track has a clear side -- establish a
                # baseline without counting a crossing.
                self._committed_side[track_id] = candidate_side
                continue

            if candidate_side != prev_side:
                if prev_side == "above" and candidate_side == "below":
                    self.count_out += 1
                elif prev_side == "below" and candidate_side == "above":
                    self.count_in += 1
                self._committed_side[track_id] = candidate_side
            # else: same side as before, nothing changed -- no-op.
