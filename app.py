# Task 4 web application

import os
import tempfile

import cv2
import gradio as gr

from detector import YoloDetector
from tracker.sort import Sort
from analytics import TrajectoryManager, ObjectCounter, LineCounter
from visualization import (
    draw_box,
    draw_trail,
    draw_counting_line,
    draw_dashboard,
)


MODEL_PATH = "yolov8n.pt"


def process_video(video_path, confidence, line_position):
    if video_path is None:
        return None, "Please upload a video."

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, "Could not open the uploaded video."

    detector = YoloDetector(
        MODEL_PATH,
        conf=float(confidence)
    )

    tracker = Sort(
        max_age=20,
        min_hits=3,
        iou_threshold=0.3
    )

    trajectories = TrajectoryManager(
        trail_length=30,
        ttl_frames=20
    )

    counter = ObjectCounter()

    line_counter = None
    if line_position is not None:
        line_counter = LineCounter(float(line_position))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    output_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    )
    output_path = output_file.name
    output_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_idx += 1

            # YOLO detection
            detections = detector.detect(frame)

            # SORT tracking
            tracks = tracker.update(
                detections.boxes,
                detections.scores,
                detections.class_ids
            )

            active_for_trails = []
            active_for_counter = []
            active_for_line = []

            for x1, y1, x2, y2, track_id, class_id, score in tracks:

                track_id = int(track_id)
                class_id = int(class_id)

                class_name = detector.class_name(class_id)

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                draw_box(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    track_id,
                    class_name,
                    score
                )

                active_for_trails.append(
                    (track_id, cx, cy)
                )

                active_for_counter.append(
                    (track_id, class_name)
                )

                active_for_line.append(
                    (track_id, cy)
                )

            # Trajectories
            trajectories.update(
                frame_idx,
                active_for_trails
            )

            for track_id, _, _ in active_for_trails:
                draw_trail(
                    frame,
                    track_id,
                    trajectories.get_trail(track_id)
                )

            # Object counting
            counts = counter.update(
                active_for_counter
            )

            count_in = 0
            count_out = 0

            # Line crossing
            if line_counter is not None:

                line_counter.update(
                    height,
                    active_for_line
                )

                count_in = line_counter.count_in
                count_out = line_counter.count_out

                draw_counting_line(
                    frame,
                    line_counter.line_pixel_y(height),
                    count_in,
                    count_out
                )

            # Dashboard
            draw_dashboard(
                frame,
                0.0,
                "SORT",
                counts,
                count_in,
                count_out,
                line_enabled=line_counter is not None
            )

            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    summary = (
        f"Processing complete\n"
        f"Total unique objects tracked: "
        f"{counts.get('total_unique_tracked', 0)}\n"
        f"IN: {count_in}\n"
        f"OUT: {count_out}"
    )

    return output_path, summary


with gr.Blocks(
    title="CodeAlpha Task 4 - Object Detection & Tracking"
) as demo:

    gr.Markdown(
        """
        # 🚀 Real-Time Object Detection & Tracking

        **CodeAlpha Internship Task 4**

        YOLOv8 + SORT + Trajectory Tracking + Object Counting
        """
    )

    gr.Markdown(
        """
        Upload a video and the system will detect and track objects,
        assign tracking IDs, draw motion trails, count objects,
        and optionally perform IN/OUT line crossing.
        """
    )

    with gr.Row():

        with gr.Column():

            video_input = gr.Video(
                label="Upload Input Video"
            )

            confidence = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.4,
                step=0.05,
                label="YOLO Confidence"
            )

            line_position = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="Counting Line Position"
            )

            process_button = gr.Button(
                "▶ Run Object Detection & Tracking"
            )

        with gr.Column():

            video_output = gr.Video(
                label="Processed Output"
            )

            status_output = gr.Textbox(
                label="Analytics",
                lines=5
            )

    process_button.click(
        fn=process_video,
        inputs=[
            video_input,
            confidence,
            line_position
        ],
        outputs=[
            video_output,
            status_output
        ]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
