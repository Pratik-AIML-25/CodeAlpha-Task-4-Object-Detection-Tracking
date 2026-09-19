# Real-Time Object Detection & Tracking

**Internship Task 4** — YOLO detection + SORT/DeepSORT tracking, with trajectory trails,
object counting, and virtual line crossing, built on OpenCV.

---
🚀 Live Demo
https://pratik-aiml-25.github.io/CodeAlpha-Task-4-Object-Detection-Tracking/

## 1. Problem Statement

Detect objects in a live video stream (webcam or file) and give each one a **persistent
identity** across frames — not just "there is a person here," but "this is the *same*
person seen 40 frames ago." Detection alone answers "what and where, right now." Tracking
adds "which one, over time," which is what makes counting, trajectories, and line-crossing
possible.

## 2. Features

- Real-time detection on webcam or video file (OpenCV `VideoCapture`)
- YOLOv8 detector — swappable model size, configurable confidence, optional class filter
- **Class-aware SORT** tracker built from scratch (Kalman filter + Hungarian algorithm)
  — a person can never "become" a car mid-track, even under heavy box overlap
- Optional **DeepSORT** backend (appearance-based re-identification) via one CLI flag
- Motion trails per tracked object, capped length, auto-cleaned when a track disappears
- Live object counting: active count, per-class breakdown, total unique objects ever tracked
- Optional virtual line with IN/OUT crossing counts
- Clean on-screen dashboard (FPS, tracker in use, counts) without covering the video
- Graceful error handling for bad video sources, missing models, and missing optional
  dependencies
- A small benchmarking script that reports *measured* FPS — no invented numbers

## 3. Technology Stack

| Component | Choice |
|---|---|
| Language | Python 3.9+ |
| Video I/O & drawing | OpenCV |
| Detector | Ultralytics YOLOv8 (`yolov8n.pt` default) |
| Tracker (default) | Custom SORT — `filterpy` Kalman filter + `scipy` Hungarian algorithm |
| Tracker (optional) | DeepSORT via `deep-sort-realtime` |
| Core numerics | NumPy |

## 4. Architecture / Pipeline

```
Video Input (webcam / file)
        |
        v
     OpenCV  (frame capture)
        |
        v
  YOLO Detection            <-- detector.py
        |
        v
  Bounding Boxes + Scores + Class IDs
        |
        v
  SORT  /  DeepSORT          <-- tracker/sort.py, tracker/deepsort_tracker.py
        |
        v
  Stable Tracking IDs (class-aware)
        |
        v
  Trajectory Trails + Object Counting + Line Crossing   <-- analytics.py
        |
        v
  Dashboard / Overlay Drawing                            <-- visualization.py
        |
        v
  Display window  +  optional saved output video
```

`main.py` only wires these pieces together frame by frame — it doesn't contain detection,
tracking, or drawing logic itself, which keeps it short and easy to explain.

## 5. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`deep-sort-realtime` is optional (only needed for `--tracker deepsort`); everything else in
`requirements.txt` is required. The first run of YOLO auto-downloads `yolov8n.pt` (~6 MB).

## 6. Usage

```bash
python main.py --source 0                      # webcam
python main.py --source video.mp4               # video file
python main.py --source 0 --tracker deepsort     # DeepSORT instead of SORT
python main.py --source 0 --classes 0 2          # only track persons (0) and cars (2)
python main.py --source 0 --line 0.5             # enable IN/OUT counting at mid-height
python main.py --source video.mp4 --output outputs/demo.mp4
```

Press **`q`** in the video window to quit.

### 6a. Webcam example
```bash
python main.py --source 0 --tracker sort --line 0.6
```
Opens the default webcam, tracks everything YOLO recognizes, and counts crossings of a
line placed at 60% of the frame height.

### 6b. Video-file example
```bash
python main.py --source samples/street.mp4 --classes 0 2 3 5 7 --output outputs/street_out.mp4
```
Tracks people, cars, motorcycles, buses and trucks (COCO ids) in a saved clip and writes an
annotated copy to `outputs/street_out.mp4`.

## 7. SORT — how it actually works here

1. **Predict**: every existing track has a Kalman filter over
   `[center_x, center_y, area, aspect_ratio]` with a constant-velocity model. Each frame,
   it predicts where the box should be *before* seeing the new detections.
2. **Associate (class-aware)**: predicted boxes are compared to this frame's detections
   using **IoU**. Matching is done **separately per class** — persons only match against
   person tracks, cars only against car tracks — then results are merged. This is the fix
   for the classic "Person ID 3 becomes Car ID 3" bug when boxes overlap heavily.
3. **Optimal assignment**: within each class, the **Hungarian algorithm**
   (`scipy.optimize.linear_sum_assignment`) finds the matching that maximizes total IoU,
   rather than greedily grabbing the first decent match.
4. **Update / create / delete**: matched tracks get their Kalman filter corrected with the
   real detection. Unmatched detections start new tracks. Tracks unmatched for more than
   `--max-age` frames are dropped. A track is only shown once it accumulates `--min-hits`
   consecutive detections (filters out one-frame false positives).

See `tracker/sort.py` — every function is commented at the level of "what math is this and
why," not "what does this line of Python do," so it holds up in a viva.

## 8. DeepSORT — how it differs

DeepSORT keeps SORT's Kalman filter + IoU idea but adds a small CNN **appearance embedding**
for each detected crop. Matching then blends motion distance with appearance similarity, so
two people who briefly cross paths (and would confuse IoU-only matching) are less likely to
swap IDs. The cost is extra compute per frame (running a small embedding network on every
box). Switch with `--tracker deepsort`; if `deep-sort-realtime` isn't installed, the app
prints a clear warning and **falls back to SORT** instead of crashing.

| | SORT | DeepSORT |
|---|---|---|
| Motion model | Kalman filter | Kalman filter |
| Association | IoU + Hungarian algorithm | IoU/Mahalanobis + appearance embedding |
| Speed | Fast | Slower (extra CNN pass per box) |
| Occlusion / ID switches | More prone to switches | Generally more stable |
| Dependencies | `filterpy`, `scipy` | + `deep-sort-realtime` |

## 9. Object counting

`analytics.py`'s `ObjectCounter` counts by **unique track ID**, not by detection: a person
detected in 50 consecutive frames is still one object, not 50. It reports:
- `Active Objects` — tracks visible right now
- per-class active breakdown (e.g. `person: 3`, `car: 2`)
- `Total Tracked` — every unique ID seen since the stream started

## 10. Line-crossing (IN/OUT)

`LineCounter` places a horizontal line at a configurable fraction of the frame height
(`--line 0.5` = middle). A change from "above" to "below" counts **OUT**, "below" to
"above" counts **IN**.

**Hysteresis / dead-zone, not a single pixel line.** A naive version that just checks
"which side of the line is the center on right now" flips constantly when a track's center
wobbles by a pixel or two near the line -- one real crossing can register as several. Instead,
two thresholds surround the line:

```
                upper_bound = line_y - margin      (clearly ABOVE)
frame  ---------------------------------------------------------
top      ...
                dead zone  (ambiguous -- ignored)
         ---------------------  line_y  ----------------------------
                dead zone  (ambiguous -- ignored)
         ...
                lower_bound = line_y + margin      (clearly BELOW)
```

Each track has a **committed side** (above/below) that only updates once its center clearly
passes one of these thresholds. While inside the dead zone, jitter is simply ignored -- the
committed side doesn't change, so nothing is counted. A crossing is only registered when the
committed side actually flips from one clear side to the other, which both enforces
*meaningful* movement across the line and guarantees each physical crossing is counted
exactly once. The dead-zone half-width defaults to 2% of frame height
(`margin_fraction=0.02` in `LineCounter.__init__`) and can be widened for noisier tracks --
this is an internal tuning knob, not a new CLI flag, so `--line` still works exactly as
before.

A track that crosses, then genuinely moves back across the line later, is counted again in
the opposite direction -- that's a second real event, not a duplicate of the first. This
works best for a mostly one-directional flow, e.g. people walking through a doorway or cars
passing a checkpoint.

## 11. Configuration reference

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `0` | Webcam index or path to a video file |
| `--model` | `yolov8n.pt` | YOLOv8 weights (n/s/m/l/x — bigger = more accurate, slower) |
| `--tracker` | `sort` | `sort` or `deepsort` |
| `--confidence` | `0.4` | Minimum detection confidence |
| `--classes` | all | Restrict to given COCO class ids, e.g. `--classes 0 2` |
| `--max-age` | `20` | Frames a track survives with no matching detection |
| `--min-hits` | `3` | Consecutive hits before a new track is displayed |
| `--iou-thres` | `0.3` | IoU needed to match a detection to a track (SORT) |
| `--line` | off | Enables IN/OUT counting at this fraction of frame height (0-1) |
| `--trail-length` | `30` | Max points kept per motion trail |
| `--output` | off | Path to save the annotated video |

## 12. Performance considerations

- `yolov8n.pt` is the fastest/least accurate YOLOv8 variant — good default for real-time on
  CPU; step up to `yolov8s.pt`/`yolov8m.pt` for accuracy if you have a GPU.
- Class filtering (`--classes`) reduces downstream tracking work when you only care about
  specific object types.
- DeepSORT is meaningfully slower than SORT per frame (extra embedding network); use SORT
  for pure speed, DeepSORT when identity stability through occlusion matters more.
- FPS is measured with a rolling average (`utils.FPSCounter`) over the last 20 frames, so it
  doesn't jitter wildly frame to frame.

## 13. Project structure

```
object_detection_tracking/
├── main.py                    # wires the pipeline together, CLI entry point
├── detector.py                 # YOLOv8 wrapper (Detections dataclass, error handling)
├── tracker/
│   ├── sort.py                 # from-scratch, class-aware SORT
│   └── deepsort_tracker.py     # adapter around deep-sort-realtime (optional)
├── analytics.py                # trails, object counting, line crossing
├── visualization.py            # all OpenCV drawing (boxes, trails, dashboard)
├── utils.py                    # video I/O helpers, FPS counter, color palette, errors
├── benchmark.py                 # measures real FPS/processing time, no display
├── test_logic.py                # unit tests for tracker/analytics/visualization
├── requirements.txt
├── outputs/                     # suggested location for saved videos
└── README.md
```

## 14. Example output

At runtime you'll see a video window with:
- a colored box + `ID <n> | <class> <confidence>%` label per tracked object
- a short fading trail behind each moving object
- an info panel (top-left) with FPS, active tracker, live counts
- if `--line` is set, a horizontal line with a running `IN: x  OUT: y` count

(No screenshot is embedded here — generate one from your own run with `--output` and drop
it into your submission; that's a real result rather than a placeholder image.)

## 15. Limitations

- SORT has no appearance model, so IDs can still switch during long/total occlusions or when
  two same-class objects fully overlap for several frames — that's a fundamental IoU-tracker
  limitation, not a bug; DeepSORT reduces but does not eliminate this.
- Line-crossing assumes a **horizontal** line and mostly-vertical motion; diagonal/multi-line
  setups aren't implemented.
- Small hysteresis-free crossing detection means an object that lingers exactly on the line
  could, in principle, flicker a count if its center oscillates across the line pixel by
  pixel; not observed to be a practical problem in normal footage.
- Detection quality is entirely dependent on the pre-trained YOLO/COCO classes — objects
  outside COCO's 80 classes won't be detected without retraining/fine-tuning.
- No GPU-specific optimization included; CPU-only inference on `yolov8n.pt` will be slower
  than a CUDA GPU run.

## 16. Future improvements

- Add a Re-ID model swap-in point so DeepSORT's embedding network can be replaced with a
  domain-specific one.
- Support multiple/angled counting lines and zone-based (polygon) counting.
- Add a lightweight web dashboard (Flask/Streamlit) for remote monitoring.
- Export tracked trajectories/counts to CSV for offline analytics.

## 17. Testing

`test_logic.py` unit-tests the parts of this project that are original engineering (not
just calling a library):
1. A single moving object keeps one stable ID across frames.
2. Two fully-overlapping detections of **different classes** never swap or share an ID
   (the class-aware fix).
3. A track with no matching detections for more than `max_age` frames is correctly removed.
4. Trajectory trail length is capped.
5. **Line-crossing, normal downward crossing** — a track moving clearly from above the
   dead zone to clearly below it counts exactly one OUT.
6. **Line-crossing, normal upward crossing** — clearly below to clearly above counts
   exactly one IN.
7. **Line-crossing, jitter without crossing** — a track established on one side that then
   wobbles entirely inside the dead zone around the line produces zero IN/OUT events.
8. **Line-crossing, crossing then back** — a genuine OUT crossing followed later by a
   genuine return crossing counts one OUT and one IN (two real events, not a duplicate);
   small movement after the return trip that doesn't re-cross the line adds nothing further.
9. The full drawing pipeline (boxes, trail, line, dashboard) runs on a blank frame without
   error.

Run it with:
```bash
pip install numpy scipy filterpy opencv-python
python test_logic.py
```
**I executed `test_logic.py` in the assistant's sandbox and all 9 checks passed** (output:
9x `[PASS]`, ending "All self-tests passed."). **Full end-to-end runs against a live webcam
and against the real YOLOv8 model were not executed** — that sandbox has no camera and ran
out of disk space installing PyTorch (a genuine environment limitation, reported honestly
rather than papered over with invented FPS numbers). Run `python main.py --source 0` and
`python benchmark.py --source your_video.mp4` on your own machine to get real FPS numbers
for your report — `benchmark.py` prints only measured values, never estimates. If you add
`--line` to a real run, walking across the line yourself is the best manual check that IN/OUT
now stays stable instead of flickering when you pause near it.

Manual checks still recommended on your machine before submitting:

| Test | What to check |
|---|---|
| Webcam to YOLO to SORT | `python main.py --source 0` opens a window, boxes track you |
| Video file to YOLO to SORT | `python main.py --source your_video.mp4` |
| Multiple objects, stable IDs | Walk two people through frame; IDs shouldn't jump |
| Crossing objects | Have two people cross paths; note any ID switch (expected occasionally with SORT) |
| Line crossing | `--line 0.5`; walk across the line, confirm IN/OUT increments once each |
| Class-aware tracking | `--classes 0 2` near overlapping person/car; confirm no ID sharing |
| Invalid input | `python main.py --source no_such_file.mp4` should print a clear error, not a traceback |
| DeepSORT | `pip install deep-sort-realtime` then `--tracker deepsort` |

---

## VIVA QUESTIONS I MUST KNOW

**1. What's the difference between object detection and object tracking?**
Detection finds objects (box + class + confidence) independently in a single frame.
Tracking links detections across frames so the same physical object keeps one ID over time.

**2. What is a bounding box?**
A rectangle, usually given as `(x1, y1, x2, y2)` (top-left and bottom-right corners) or
`(x, y, w, h)`, that marks where an object is in an image.

**3. What is the confidence threshold and why does it matter?**
The minimum score the model must assign a detection before we keep it. Too low means lots of
false positives; too high means real objects get missed. `0.4` here is a practical middle ground.

**4. What is IoU (Intersection over Union)?**
`overlap_area / union_area` of two boxes. IoU = 1 means identical boxes, IoU = 0 means no
overlap. It's the standard way to measure "how similar are these two boxes."

**5. Why is IoU used for tracking association?**
Because a real object usually moves only a little between consecutive frames, so its new
detection strongly overlaps its previous predicted position — high IoU implies "probably the
same object."

**6. What is a Kalman Filter and why use it here?**
An algorithm that estimates a system's state (here: box center, size, aspect ratio, and
their velocities) from noisy observations, and predicts the next state before the next
measurement arrives. It smooths detection noise and lets a track survive a missed detection
for a frame or two.

**7. What does "constant velocity model" mean?**
The Kalman filter assumes each object moves at roughly constant speed between frames — a
simple but effective approximation for short time steps.

**8. What is the Hungarian algorithm doing in this project?**
Given a cost matrix (here, negative IoU between every track-prediction and every detection),
it finds the assignment that minimizes total cost (maximizes total IoU) — an optimal
one-to-one matching, better than greedily picking the first good match.

**9. What is SORT?**
Simple Online and Realtime Tracking: predict each track's next box with a Kalman filter,
match predictions to new detections via IoU + Hungarian algorithm, update matched tracks,
create new tracks for unmatched detections, delete tracks that go too long unmatched.

**10. What is DeepSORT and how is it different from SORT?**
DeepSORT adds a CNN-based appearance embedding per detection, so association considers both
motion and how similar the object *looks*, not just its position. This helps keep IDs
consistent through occlusion, at the cost of extra computation.

**11. What is `max_age` in this project?**
The number of frames a track is allowed to go without a matching detection before it's
deleted — handles brief occlusion or a missed detection without losing the object's identity.

**12. What is `min_hits`?**
The number of consecutive successful matches a new track needs before it's actually shown —
filters out short-lived false-positive detections from ever appearing as a "confirmed" track.

**13. What is an ID switch, and why does it happen?**
When a tracker mistakenly assigns one object's existing ID to a different object (or swaps
IDs between two objects) — usually happens under occlusion or when objects cross paths, since
IoU-only tracking has no idea what the objects actually look like.

**14. Why is class-aware association important?**
Without it, a track could jump to a detection of a *different class* just because their boxes
overlap a lot (e.g. a person carrying a bag). Restricting matching to same-class
detections/tracks prevents "Person ID 3 becomes Car ID 3."

**15. What is occlusion, and how does this project handle it?**
Occlusion is when an object is temporarily hidden (fully or partially) behind something else.
`max_age` lets the Kalman filter keep predicting the object's position for a few frames
without a real detection, so the track can "survive" a brief occlusion and reconnect once the
object reappears.

**16. What is trajectory tracking / motion trails used for?**
Storing an object's recent center points and drawing them as a line visually proves the
tracker is maintaining identity over time (not just re-detecting), and is a stepping stone to
path analysis (e.g. did this vehicle turn, loiter, or move in a restricted direction).

**17. How does virtual line-crossing counting work?**
Each track's center-point side of a defined line ("above"/"below") is remembered each frame;
a change of side is a crossing event, counted as IN or OUT depending on the direction —
useful for footfall/vehicle counting without needing a full zone-based system.

**18. What is FPS and why measure it here?**
Frames Per Second — how many frames the whole detect+track+draw pipeline processes per
second. It's the practical measure of whether a pipeline is "real-time" for its use case.

**19. Why is tracking needed after detection at all — why not just detect every frame?**
Detection alone gives no notion of identity: you'd know "there are 3 people" every frame but
never "person A walked from left to right." Tracking is what enables counting individuals
(not detections), trajectories, and directional analytics like line crossing.

**20. Why use a pre-trained model like YOLO instead of training your own detector?**
Pre-trained YOLO (trained on COCO, 80 common classes) already generalizes well to everyday
objects, so a project like this doesn't need thousands of labeled images or GPU-hours of
training just to detect people/cars/etc.
