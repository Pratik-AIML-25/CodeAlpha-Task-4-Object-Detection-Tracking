"""test_logic.py
Unit tests for the parts of this project that are original engineering:
class-aware SORT tracking, trajectory/counting analytics, line-crossing
hysteresis, and the drawing layer -- none of which need YOLO weights,
a GPU, or a camera, so this suite runs anywhere.

Run with:  python test_logic.py
"""

import os
import tempfile
from collections import deque

import numpy as np
import cv2

from tracker.sort import Sort
from analytics import TrajectoryManager, ObjectCounter, LineCounter
from visualization import draw_box, draw_trail, draw_counting_line, draw_dashboard

COCO_PERSON, COCO_CAR = 0, 2

def make_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_basic_tracking_and_id_stability():
    tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.3)
    # Person moving right across 10 frames -> should keep the same ID
    ids_seen = set()
    for i in range(10):
        x = 50 + i * 10
        boxes = np.array([[x, 50, x + 40, 150]])
        scores = np.array([0.9])
        classes = np.array([COCO_PERSON])
        tracks = tracker.update(boxes, scores, classes)
        assert len(tracks) == 1, f"expected 1 track, got {len(tracks)}"
        ids_seen.add(int(tracks[0][4]))
    assert len(ids_seen) == 1, f"ID should stay stable, saw {ids_seen}"
    print("[PASS] basic tracking keeps a stable ID across frames")

def test_class_aware_no_id_swap():
    tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.1)
    # A person and a car heavily overlapping in the SAME location.
    boxes = np.array([[100, 100, 200, 300], [100, 100, 200, 300]])
    scores = np.array([0.9, 0.9])
    classes = np.array([COCO_PERSON, COCO_CAR])
    tracks1 = tracker.update(boxes, scores, classes)
    assert len(tracks1) == 2
    person_id = int(tracks1[tracks1[:, 5] == COCO_PERSON][0][4])
    car_id = int(tracks1[tracks1[:, 5] == COCO_CAR][0][4])

    # Next frame: same overlapping boxes again -- classes must not swap IDs.
    tracks2 = tracker.update(boxes, scores, classes)
    person_id2 = int(tracks2[tracks2[:, 5] == COCO_PERSON][0][4])
    car_id2 = int(tracks2[tracks2[:, 5] == COCO_CAR][0][4])
    assert person_id == person_id2, "person ID changed unexpectedly"
    assert car_id == car_id2, "car ID changed unexpectedly"
    assert person_id != car_id, "person and car got the same ID"
    print("[PASS] class-aware association keeps person/car IDs separate despite full overlap")

def test_track_removed_after_max_age():
    tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.3)
    boxes = np.array([[10, 10, 50, 50]])
    scores = np.array([0.9])
    classes = np.array([COCO_PERSON])
    tracker.update(boxes, scores, classes)
    empty = np.empty((0, 4))
    for _ in range(5):
        tracks = tracker.update(empty, np.empty((0,)), np.empty((0,), dtype=int))
    assert len(tracks) == 0, "track should be dropped after max_age with no detections"
    print("[PASS] stale tracks are pruned after max_age")

def test_trajectory_and_counter():
    traj = TrajectoryManager(trail_length=5, ttl_frames=5)
    counter = ObjectCounter()

    for i in range(6):
        cy = 100 + i * 40
        traj.update(i, [(1, 300, cy)])
        counter.update([(1, "person")])

    assert len(traj.get_trail(1)) <= 5
    print("[PASS] trajectory trail length is capped")


# --- LineCounter: hysteresis / dead-zone behaviour ---
# Frame height 480, line at fraction 0.5 -> line_y = 240.
# Default margin_fraction=0.02 -> margin = max(1, 480*0.02) = 9.6px
# -> dead zone is y in (230.4, 249.6); "above" means y < 230.4, "below" means y > 249.6.

def test_line_crossing_downward():
    line = LineCounter(line_y_fraction=0.5)
    # Clearly above -> clearly below, in a handful of frames.
    ys = [100, 150, 200, 260, 300, 320]
    for cy in ys:
        line.update(frame_height=480, active_tracks=[(1, cy)])
    assert line.count_out == 1, f"expected exactly 1 OUT, got {line.count_out}"
    assert line.count_in == 0, f"expected 0 IN, got {line.count_in}"
    print("[PASS] normal downward crossing counts exactly one OUT")


def test_line_crossing_upward():
    line = LineCounter(line_y_fraction=0.5)
    # Clearly below -> clearly above.
    ys = [320, 300, 260, 200, 150, 100]
    for cy in ys:
        line.update(frame_height=480, active_tracks=[(1, cy)])
    assert line.count_in == 1, f"expected exactly 1 IN, got {line.count_in}"
    assert line.count_out == 0, f"expected 0 OUT, got {line.count_out}"
    print("[PASS] normal upward crossing counts exactly one IN")


def test_line_jitter_no_crossing():
    line = LineCounter(line_y_fraction=0.5)
    # Establish a clear starting side first.
    line.update(frame_height=480, active_tracks=[(1, 150)])  # clearly above
    # Now wobble entirely inside the dead zone (230.4, 249.6) around the line.
    jitter_ys = [235, 244, 238, 247, 233, 241, 236, 245]
    for cy in jitter_ys:
        line.update(frame_height=480, active_tracks=[(1, cy)])
    assert line.count_in == 0 and line.count_out == 0, (
        f"jitter inside the dead zone should never count, got IN={line.count_in} OUT={line.count_out}"
    )
    print("[PASS] jitter confined to the dead zone produces zero false crossings")


def test_line_crossing_then_back():
    line = LineCounter(line_y_fraction=0.5)
    # Cross downward (OUT)...
    for cy in [100, 150, 200, 260, 300]:
        line.update(frame_height=480, active_tracks=[(1, cy)])
    assert line.count_out == 1 and line.count_in == 0

    # ...then cross back upward (IN). This is a second, distinct physical
    # crossing and must be counted -- it is not the same event being
    # double-counted.
    for cy in [300, 260, 200, 150, 100]:
        line.update(frame_height=480, active_tracks=[(1, cy)])
    assert line.count_out == 1, f"OUT should still be 1, got {line.count_out}"
    assert line.count_in == 1, f"expected the return trip to register 1 IN, got {line.count_in}"

    # A tiny wobble back toward (but not through) the line afterwards must
    # not add any further counts.
    for cy in [140, 148, 142]:
        line.update(frame_height=480, active_tracks=[(1, cy)])
    assert line.count_out == 1 and line.count_in == 1, (
        "small movement after the return crossing must not add extra counts"
    )
    print("[PASS] crossing followed by a genuine return crossing counts one OUT and one IN, no more")

def test_drawing_does_not_crash():
    frame = make_frame()
    h, w = frame.shape[:2]

    # Each draw_* must return the SAME frame object it was given (in-place
    # contract), not a copy and not None.
    returned = draw_box(frame, 10, 10, 100, 100, track_id=1, class_name="person", score=0.87)
    assert returned is frame, "draw_box must return the frame it drew on"
    assert frame.sum() > 0, "draw_box should have drawn a visible box"

    trail = deque([(10, 10), (20, 20), (30, 30)])
    assert draw_trail(frame, 1, trail) is frame, "draw_trail must return the frame"

    assert draw_counting_line(frame, 240, count_in=3, count_out=2) is frame, \
        "draw_counting_line must return the frame"

    counts = {"active_total": 2, "active_by_class": {"person": 1, "car": 1}, "total_unique_tracked": 5}
    assert draw_dashboard(frame, fps=24.3, tracker_name="sort", counts=counts,
                          count_in=3, count_out=2, line_enabled=True) is frame, \
        "draw_dashboard must return the frame"

    # Frame must keep its shape/dtype -- drawing must never reallocate it.
    assert frame.shape == (h, w, 3) and frame.dtype == np.uint8

    # Write the sample frame somewhere that exists on every OS. Using
    # tempfile.gettempdir() instead of a hardcoded POSIX "/tmp" -- the
    # latter does not exist on Windows, where cv2.imwrite silently returns
    # False rather than raising.
    out_path = os.path.join(tempfile.gettempdir(), "_selftest_frame.png")
    ok = cv2.imwrite(out_path, frame)
    assert ok, f"cv2.imwrite failed to write {out_path}"
    assert os.path.exists(out_path), f"{out_path} was reported written but does not exist"
    os.remove(out_path)
    print("[PASS] drawing pipeline runs, returns the frame, and writes a sample image")


def test_drawing_with_no_tracks():
    """The overlay must render fine on a frame with zero tracked objects --
    no boxes, no trails, empty per-class breakdown."""
    frame = make_frame()

    # A trail with too few points, and an explicitly empty one, are both no-ops.
    assert draw_trail(frame, 1, deque([(10, 10)])) is frame
    assert draw_trail(frame, 1, deque()) is frame
    assert draw_trail(frame, 1, None) is frame
    assert frame.sum() == 0, "no drawing should have happened yet"

    empty_counts = {"active_total": 0, "active_by_class": {}, "total_unique_tracked": 0}
    result = draw_dashboard(frame, fps=0.0, tracker_name="sort", counts=empty_counts,
                            count_in=0, count_out=0, line_enabled=False)
    assert result is frame
    assert frame.sum() > 0, "the dashboard panel should still render with zero tracks"
    print("[PASS] drawing handles the empty/no-track case without crashing")

if __name__ == "__main__":
    tests = [
        test_basic_tracking_and_id_stability,
        test_class_aware_no_id_swap,
        test_track_removed_after_max_age,
        test_trajectory_and_counter,
        test_line_crossing_downward,
        test_line_crossing_upward,
        test_line_jitter_no_crossing,
        test_line_crossing_then_back,
        test_drawing_does_not_crash,
        test_drawing_with_no_tracks,
    ]

    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {test.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {test.__name__}: {type(exc).__name__}: {exc}")

    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    raise SystemExit(1 if failed else 0)
