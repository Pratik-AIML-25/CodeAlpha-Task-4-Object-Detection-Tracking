"""
main.py
-------
TASK 4: Object Detection and Tracking

Pipeline:
    Video Input -> OpenCV -> YOLO Detection -> Bounding Boxes ->
    SORT / DeepSORT -> Tracking IDs -> Trajectory + Counting ->
    Analytics Overlay -> Display / Output Video

Run `python main.py --help` for all options, or see README.md.
"""

import argparse
import sys

import cv2

from analytics import LineCounter, ObjectCounter, TrajectoryManager
from detector import YoloDetector
from tracker.deepsort_tracker import DeepSortTracker, DeepSortUnavailableError
from tracker.sort import Sort
from utils import FPSCounter, ModelLoadError, VideoSourceError, make_video_writer, open_video_source
from visualization import draw_box, draw_counting_line, draw_dashboard, draw_trail


def parse_args():
    p = argparse.ArgumentParser(description="Real-time object detection and tracking")
    p.add_argument("--source", default="0", help="Webcam index (e.g. 0) or path to a video file")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO weights (yolov8n/s/m/l/x.pt)")
    p.add_argument("--tracker", choices=["sort", "deepsort"], default="sort")
    p.add_argument("--confidence", type=float, default=0.4, help="Detection confidence threshold")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                    help="Restrict detection to given COCO class ids, e.g. --classes 0 2 (person, car)")
    p.add_argument("--max-age", dest="max_age", type=int, default=20,
                    help="Frames a track survives without a matching detection")
    p.add_argument("--min-hits", dest="min_hits", type=int, default=3,
                    help="Consecutive hits needed before a new track is displayed")
    p.add_argument("--iou-thres", dest="iou_thres", type=float, default=0.3,
                    help="IoU threshold for matching detections to tracks (SORT only)")
    p.add_argument("--line", type=float, default=None,
                    help="Enable IN/OUT counting with a horizontal line at this fraction "
                         "of the frame height, e.g. --line 0.5")
    p.add_argument("--trail-length", type=int, default=30, help="Max points kept per motion trail")
    p.add_argument("--output", default=None, help="Path to save annotated output video (e.g. outputs/demo.mp4)")
    return p.parse_args()


def build_tracker(name: str, max_age: int, min_hits: int, iou_thres: float):
    if name == "sort":
        return Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_thres)

    try:
        return DeepSortTracker(max_age=max_age)
    except DeepSortUnavailableError as exc:
        print(f"[WARN] {exc}")
        print("[WARN] Falling back to SORT.")
        return Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_thres)


def run(args) -> int:
    try:
        cap = open_video_source(args.source)
    except VideoSourceError as exc:
        print(f"[ERROR] {exc}")
        return 1

    try:
        detector = YoloDetector(args.model, conf=args.confidence, classes=args.classes)
    except ModelLoadError as exc:
        print(f"[ERROR] {exc}")
        cap.release()
        return 1

    tracker = build_tracker(args.tracker, args.max_age, args.min_hits, args.iou_thres)
    tracker_name = "sort" if isinstance(tracker, Sort) else "deepsort"

    fps_counter = FPSCounter()
    trajectories = TrajectoryManager(trail_length=args.trail_length, ttl_frames=args.max_age)
    counter = ObjectCounter()
    line_counter = LineCounter(args.line) if args.line is not None else None

    writer = None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if args.output:
        try:
            writer = make_video_writer(args.output, source_fps, (width, height))
        except VideoSourceError as exc:
            print(f"[ERROR] {exc}")
            cap.release()
            return 1

    print(f"[INFO] Source: {args.source} | Model: {args.model} | Tracker: {tracker_name}")
    print("[INFO] Press 'q' to quit.")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                print("[INFO] End of stream or empty frame.")
                break
            frame_idx += 1

            detections = detector.detect(frame)

            if isinstance(tracker, Sort):
                tracks = tracker.update(detections.boxes, detections.scores, detections.class_ids)
            else:
                tracks = tracker.update(detections.boxes, detections.scores, detections.class_ids, frame=frame)

            active_for_trails = []
            active_for_counter = []
            active_for_line = []

            for x1, y1, x2, y2, track_id, class_id, score in tracks:
                track_id, class_id = int(track_id), int(class_id)
                class_name = detector.class_name(class_id)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                draw_box(frame, x1, y1, x2, y2, track_id, class_name, score)
                active_for_trails.append((track_id, cx, cy))
                active_for_counter.append((track_id, class_name))
                active_for_line.append((track_id, cy))

            trajectories.update(frame_idx, active_for_trails)
            for track_id, _, _ in active_for_trails:
                draw_trail(frame, track_id, trajectories.get_trail(track_id))

            counts = counter.update(active_for_counter)

            count_in = count_out = 0
            if line_counter is not None:
                line_counter.update(height, active_for_line)
                count_in, count_out = line_counter.count_in, line_counter.count_out
                draw_counting_line(frame, line_counter.line_pixel_y(height), count_in, count_out)

            fps = fps_counter.tick()
            draw_dashboard(frame, fps, tracker_name, counts, count_in, count_out,
                            line_enabled=line_counter is not None)

            cv2.imshow("Object Detection & Tracking", frame)
            if writer is not None:
                writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit requested by user.")
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args()))
