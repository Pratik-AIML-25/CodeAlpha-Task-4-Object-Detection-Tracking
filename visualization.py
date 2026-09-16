"""
visualization.py
-----------------
All OpenCV drawing lives here so main.py stays focused on the pipeline
logic. The video frame itself is always left clearly visible -- the info
panel is a small, semi-transparent overlay in the corner.

API contract
------------
Every draw_* function mutates the frame in place (standard OpenCV
behaviour, and what main.py relies on) AND returns that same frame object.
Returning it makes the contract explicit and allows chaining, e.g.

    draw_dashboard(draw_counting_line(frame, ...), ...)

The returned array is always the same object that was passed in -- never a
copy -- so callers that ignore the return value behave exactly as before.
"""

from collections import deque
from typing import Optional, Sequence

import cv2
import numpy as np

from utils import get_color


def draw_box(frame: np.ndarray, x1, y1, x2, y2, track_id: int,
             class_name: str, score: float) -> np.ndarray:
    """Draws one tracked object's box with an "ID | class conf%" label.
    Returns the same frame, drawn on in place."""
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    color = get_color(track_id)
    label = f"ID {track_id} | {class_name} {score * 100:.0f}%"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def draw_trail(frame: np.ndarray, track_id: int,
               points: Optional[Sequence] = None) -> np.ndarray:
    """Draws a motion trail behind a tracked object's center point.
    A trail of fewer than 2 points has nothing to draw, so the frame is
    returned untouched. Returns the same frame."""
    if points is None or len(points) < 2:
        return frame
    color = get_color(track_id)
    pts = list(points)
    for i in range(1, len(pts)):
        cv2.line(frame, tuple(map(int, pts[i - 1])), tuple(map(int, pts[i])), color, 2)
    return frame


def draw_counting_line(frame: np.ndarray, line_y: int,
                       count_in: int, count_out: int) -> np.ndarray:
    """Draws the horizontal counting line plus its IN/OUT tally.
    Returns the same frame."""
    width = frame.shape[1]
    line_y = int(line_y)
    cv2.line(frame, (0, line_y), (width, line_y), (0, 200, 255), 2)
    cv2.putText(frame, f"IN: {count_in}  OUT: {count_out}", (width // 2 - 90, max(20, line_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return frame


def draw_dashboard(frame: np.ndarray, fps: float, tracker_name: str, counts: dict,
                   count_in: int, count_out: int, line_enabled: bool) -> np.ndarray:
    """Draws a semi-transparent info panel in the top-left corner.
    Handles the empty/no-track case (zero active objects, empty per-class
    breakdown) without special-casing by the caller. Returns the same frame."""
    active_by_class = counts.get("active_by_class") or {}

    lines = [
        f"FPS: {fps:.1f}",
        f"Tracker: {tracker_name.upper()}",
        f"Active Objects: {counts.get('active_total', 0)}",
    ]
    for class_name, n in sorted(active_by_class.items()):
        lines.append(f"  {class_name}: {n}")
    lines.append(f"Total Tracked: {counts.get('total_unique_tracked', 0)}")
    if line_enabled:
        lines.append(f"IN: {count_in}  OUT: {count_out}")

    panel_w = 260
    panel_h = 28 * len(lines) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 34
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return frame
