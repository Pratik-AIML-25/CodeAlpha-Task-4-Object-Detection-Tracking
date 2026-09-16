"""
benchmark.py
------------
Measures *actual* performance of the detect+track pipeline on a given
source -- no numbers here are invented. Runs with no GUI window so display
overhead doesn't skew the measurement.

Usage:
    python benchmark.py --source video.mp4 --frames 200
    python benchmark.py --source 0 --tracker deepsort --frames 100
"""

import argparse
import time

from detector import YoloDetector
from main import build_tracker
from tracker.sort import Sort
from utils import ModelLoadError, VideoSourceError, open_video_source


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark detection+tracking FPS")
    p.add_argument("--source", default="0")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--tracker", choices=["sort", "deepsort"], default="sort")
    p.add_argument("--confidence", type=float, default=0.4)
    p.add_argument("--frames", type=int, default=150, help="Number of frames to benchmark")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        cap = open_video_source(args.source)
    except VideoSourceError as exc:
        print(f"[ERROR] {exc}")
        return

    try:
        detector = YoloDetector(args.model, conf=args.confidence)
    except ModelLoadError as exc:
        print(f"[ERROR] {exc}")
        cap.release()
        return

    tracker = build_tracker(args.tracker, max_age=20, min_hits=3, iou_thres=0.3)
    tracker_name = "sort" if isinstance(tracker, Sort) else "deepsort"

    processed = 0
    start = time.time()

    while processed < args.frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[INFO] Source ended before reaching --frames.")
            break

        detections = detector.detect(frame)
        if isinstance(tracker, Sort):
            tracker.update(detections.boxes, detections.scores, detections.class_ids)
        else:
            tracker.update(detections.boxes, detections.scores, detections.class_ids, frame=frame)

        processed += 1

    elapsed = time.time() - start
    cap.release()

    print("\n--- Benchmark result (measured, not estimated) ---")
    print(f"Tracker:          {tracker_name}")
    print(f"Model:            {args.model}")
    print(f"Frames processed: {processed}")
    print(f"Total time:       {elapsed:.2f} s")
    if processed and elapsed > 0:
        print(f"Average FPS:      {processed / elapsed:.2f}")
    else:
        print("Average FPS:      N/A (no frames processed)")


if __name__ == "__main__":
    main()
