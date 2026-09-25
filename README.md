# Driver Drowsiness Detection and Alert System

A real-time, webcam-based system that watches the driver's face with **MediaPipe**
facial landmarks, computes the **Eye Aspect Ratio (EAR)** and **Mouth Aspect
Ratio (MAR)**, applies **temporal state logic** to distinguish normal blinks from
dangerous micro-sleeps, and raises a **loud audio + visual alarm** the moment
drowsiness is confirmed.

Built as a clean, modular, test-covered project — suitable for a final-year
project, portfolio/GitHub, or a technical interview demonstration.

---

## Features

* **Real-time webcam pipeline** at ~30 FPS on CPU (no GPU required).
* **MediaPipe Face Mesh** landmark detection (468-point model), with the modern
  `mediapipe.tasks` backend optionally supported.
* **EAR (Eye Aspect Ratio)** — classic blink/micro-sleep metric.
* **MAR (Mouth Aspect Ratio)** — yawn detection.
* **PERCLOS (Percentage of Eye Closure)** — rolling time window measuring the
  share of recent *time* the eyes are substantially closed: an additive
  drowsiness signal alongside the instantaneous EAR alarm.
* **Temporal state machine** with hysteresis and consecutive-frame counters:
  one short blink will *never* trigger an alarm, a sustained eye closure will.
* **Blink counter + rolling blink rate** (blinks/minute over the last minute)
  — a statistics-driven drowsiness tell.
* **Configurable thresholds** — via `src/config/settings.py` or the CLI, so the
  system can be calibrated per driver and per camera.
* **Loud siren** (synthesized, cross-platform via `pygame`, with a Windows
  `winsound` fallback) plus a full-screen visual alert.
* **Optional machine-learning extension**: feature extraction, session recording
  and a trainable Random Forest classifier that fuses with the CV rules.
* **Session logging** to CSV (production data for the ML pipeline).
* **Unit-tested** core (`pytest`) — the math and state logic run on synthetic
  landmark sequences, no webcam needed.

---

## Quickstart

### 1. Install

Requires **Python 3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # core
pip install -r requirements-dev.txt
```

### 2. Run with your webcam

```bash
python app.py
```

The OpenCV window opens showing your face with the eye/mouth regions overlaid,
live EAR/MAR, counters and the current state. **Close your eyes for ~1 second**
(or open your mouth wide for ~1 second) to see the alarm fire.

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `r` | Reset the session counters |

### 3. Useful CLI options

```bash
python app.py --source 0                        # explicit webcam index
python app.py --source videos/drive.mp4         # offline video file
python app.py --ear-threshold 0.23 --ear-frames 36
python app.py --mar-threshold 0.6
python app.py --perclos-window 60 --perclos-warning 40 --perclos-drowsy 60
python app.py --head-pitch-down 20 --head-duration 5     # head-pose support
python app.py --no-head-pose                             # disable head pose
python app.py --no-audio                        # silent mode (visual only)
python app.py --no-window                       # headless (log/metrics only)
python app.py --save-log data/sessions/run1.csv
python app.py --ml                              # enable ML classifier fusion
python app.py --detector tasks                  # MediaPipe Tasks backend (auto-downloads the model on first use)
python app.py --level DEBUG
```

---

## How it works

### The computer-vision pipeline

```
Webcam ──► BGR frame ──► RGB ──► MediaPipe FaceMesh
                                          │  468 landmarks (x, y normalized)
                                          ▼
                        ┌─────────────────────────────────┐
                        │ Eye landmark extraction (6 pts) │  EAR per eye
                        │ Mouth landmark extraction (4 pts)│ MAR
                        │ Head-pose landmarks (6 pts)      │ pitch/yaw/roll
                        └─────────────────────────────────┘
                                          ▼
Temporal state machine
                  (eye-closed, PERCLOS window, mouth-open, blink, yawn)
                                          ▼
                         Drowsiness classification ──► alarm (audio + visual)
                Head pose (smoothed + sustained) ──► on-screen reminder only
```

Every step is its own module:

| Module | Responsibility |
|--------|----------------|
| `src/camera.py` | Video capture (webcam or file) |
| `src/vision/detector.py` | Face landmark backends (+ factory) |
| `src/vision/landmarks.py` | Canonical MediaPipe landmark indices |
| `src/vision/renderer.py` | OpenCV overlays |
| `src/dynamics/metrics.py` | EAR / MAR / geometry math (pure NumPy) |
| `src/dynamics/perclos.py` | Rolling-window PERCLOS calculator |
| `src/dynamics/headpose.py` | Head-pose Euler angles, direction + temporal tracker (pure) |
| `src/vision/headpose.py` | OpenCV `solvePnP` head-pose estimator |
| `src/dynamics/state_machine.py` | Temporal counters, blinks, yawns, hysteresis, PERCLOS fusion |
| `src/alerts/*` | Audio + visual + coordinator |
| `src/ml/*` | Optional feature extraction + classifier |
| `src/pipeline.py` | End-to-end orchestration |

### Eye Aspect Ratio (EAR)

For a six-point eye layout (p₁ = outer corner, p₄ = inner corner, p₂/p₃ top lid,
p₅/p₆ bottom lid):

```
        p2 _____ p3
          \     /
p1 ------->  EYE  <-------- p4
          /     \
        p6 ------ p5

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

* **Open eye**  →  EAR ≈ `0.28 – 0.35`  (vertical eye height is significant)
* **Closed eye** →  EAR ≈ `0.05 – 0.15` (vertical collapses, width barely changes)

**Why thresholds must be calibrated**: eyelid shape, glasses, eye size and
camera distance shift EAR for different people. The same EAR value can mean
"wide open" for one driver and "heavy lid" for another. Defaults
(`ear_threshold = 0.25`, `ear_consecutive_frames = 30` ≈ 1 s at 30 FPS) are
chosen conservatively but should be tuned — see *Calibration*.

### Mouth Aspect Ratio (MAR)

```
MAR = ||p_upper - p_lower|| / ||p_left_corner - p_right_corner||

  closed mouth:  MAR ≈ 0.05 – 0.2
  yawn:          MAR ≥ 0.6
```

A short mouth open (talking, sneeze) is **not** a yawn: the state machine also
requires the mouth to stay open for `mar_consecutive_frames` frames **or**
`yawn_duration_seconds` seconds of wall-clock time — whichever is reached first —
so detection does not depend on the camera FPS. A confirmed yawn raises the
**same loud siren** as the drowsiness alarm. Persistent open-mouth periods are
separately counted as yawning events.

### PERCLOS (Percentage of Eye Closure)

**What it is.** PERCLOS is the percentage of *time* — over the last
`perclos_window_seconds` — during which the eyes are substantially closed
(`EAR < perclos_ear_threshold`). A typical alert driver's eyes are closed only
~5–15% of a minute (blinks); heavy eyelid droop or micro-sleeps push that
fraction sharply upward.

**Why it is useful.** Instantaneous EAR says *now*; PERCLOS says *recently*.
It is the classic cumulative measure of eyelid droop: it catches a driver whose
eyes keep nearly closing even when no single closure lasts the full 1-second
streak needed by the EAR alarm.

**Why a window is required.** A single blink would otherwise look like falling
asleep. A sliding time window averages out short, benign closures while keeping
persistent droop visible.

**How it is calculated.** Every *valid* frame (`EAR` present and finite)
updates a rolling `(timestamp, EAR)` window. Each sample contributes — as
closed or open — the measured time until the next valid sample:

```
PERCLOS = (closed-eye time in window) / (valid monitoring time in window) × 100
```

Time is derived from **real timestamps**, not a fixed 30 FPS assumption, so
variable-FPS cameras and dropped frames do not distort the ratio. Gaps longer
than ~1 s (e.g. the driver looks away) and invalid EAR samples are **not**
counted as monitoring time — they are never counted as closed eyes either.

**How it integrates.** PERCLOS is shown on the preview and written to the
session CSV. Above `perclos_warning_threshold` an on-screen `PERCLOS HIGH`
warning appears (visual only — a single-frame change never triggers audio).
Above `perclos_drowsy_threshold`, **sustained** for `perclos_drowsy_seconds`,
it can push the state machine into DROWSY and raise the *same siren* as the
EAR alarm (still subject to the alert cooldown). It is purely **additive**: the
existing eye-closure (EAR streak) alarm is completely unchanged.

**Limitations.** PERCLOS alone does **not** prove a driver is drowsy — it is
one signal in a multi-signal monitoring system (EAR, PERCLOS, MAR/yawn, blink
rate). The defaults (60 s window, 40% warning, 60% drowsy sustained for 15 s,
closure threshold = the main `ear_threshold`) are deliberately conservative and
**must be validated/calibrated per driver and camera**; they are not a certified
fatigue measurement.

| CLI argument | Default | Meaning |
|---|---|---|
| `--perclos-window` | `60.0` s | Sliding time window for the ratio |
| `--perclos-ear-threshold` | `ear_threshold` | EAR below which "closed" for PERCLOS |
| `--perclos-warning` | `40.0` % | On-screen warning above this |
| `--perclos-drowsy` | `60.0` % | Sustained high PERCLOS may enter DROWSY |
| `--perclos-confirm` | `15.0` s | How long PERCLOS must stay above `--perclos-drowsy` |
| `--no-perclos` | – | Disable PERCLOS entirely (EAR-only behavior) |

### Head pose (pitch / yaw / roll)

**What it is.** Every frame, six facial landmarks (nose tip, chin, the two eye
corners and the two mouth corners) are fitted to a canonical rigid 3-D face
model with OpenCV's `solvePnP`. The recovered rotation matrix is decomposed
into three angles (degrees, see `src/dynamics/headpose.py` for the exact
convention, validated by synthetic round-trip tests):

```
pitch > 0  →  nodding DOWN (chin toward the chest)   pitch < 0 → tipped back
yaw   > 0  →  turned to the driver's own LEFT        yaw   < 0 → RIGHT
roll  > 0  →  tilted toward the driver's own LEFT    roll  < 0 → RIGHT
```

The raw angles pass through a small exponential moving average
(`head_pose_smoothing_alpha`, default `0.4`) to remove landmark jitter.

**Why it is useful.** The classic early sign of falling asleep at the wheel is
the head slowly dropping forward — a pitch-down that no eye/mouth metric can
see. Head pose is the *supporting* signal that catches exactly that posture.

**How it integrates — display only, never an alarm.** Head pose is deliberately
kept OUT of the state machine. A deviation held across the two threshold axes
(`--head-pitch-down`, `--head-pitch-up`, `--head-yaw-left`,
`--head-yaw-right`) for at least `--head-duration` seconds draws a neutral
yellow on-screen `HEAD DOWN - CHECK POSTURE` reminder. It never raises the
siren and never pushes the state into DROWSY: an open-eyed driver checking the
right-hand mirror for 2 seconds must never induce a false alarm. The smoothed
angles, the current direction and the held duration are shown on the overlay
and written to the session CSV. A feed gap longer than ~1 s (face lost) resets
the held-duration counter, so a covered camera is never counted as "head
held still".

| CLI argument | Default | Meaning |
|---|---|---|
| `--head-pitch-down` | `20.0` ° | Pitch above which the head counts as nodding down |
| `--head-pitch-up` | `20.0` ° | Negative pitch below which it counts as tipped back |
| `--head-yaw-left` | `25.0` ° | Yaw above which a turn counts as LEFT |
| `--head-yaw-right` | `25.0` ° | Negative yaw below which it counts as RIGHT |
| `--head-duration` | `5.0` s | How long an abnormal pose must be held before the reminder |
| `--no-head-pose` | – | Disable head pose entirely |

**Limitations.** A single uncalibrated webcam cannot give *accurate* absolute
degrees — the camera matrix and focal length are estimated (`focal ≈ frame
width`). The *signs* and relative magnitudes are reliable, so the thresholds
must be read as "how far the driver must deviate in that direction to register"
more than as exact degrees. Tune them per driver and camera mount: a small
10-degree-looking-down threshold will trip on every glance at the speedometer.

### Temporal detection (why it matters)

Metrics on a single frame are meaningless — your eyes are literally closed for
a frame every time you blink. The state machine (`state_machine.py`) therefore:

1. increments an **eye-closed counter** while `EAR < ear_threshold`;
2. resets it the instant the eyes re-open;
3. **only** raises DROWSY when the counter exceeds `ear_consecutive_frames`;
4. scores a **blink** for short closures, tracking the rolling blink rate;
5. uses **hysteresis** (`recovery_frames`) so the alarm starts exactly once and
   stops only after the driver demonstrably wakes up — no flickering.

```
EAR │
 0.3│ ██   ██    █████   ██      ██            ███  ██
    │ ██▄▄▄██▄▄▄▄█████▄▄██▄▄▄▄▄▄██▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄██▄▄▄▄██   ...
 0.1│                  └── blink ──┘        └── 1 s CLOSED ──┘
 0.0│                                                    └────► DROWSY
```

---

## Configuring the system

Everything lives in [src/config/settings.py](src/config/settings.py), split into
focused dataclasses: `CameraConfig`, `DetectionSettings`, `AlertSettings`,
`MLConfig`. Tune either directly:

```python
from src.config.settings import Settings, DetectionSettings

settings = Settings(
    detection=DetectionSettings(ear_threshold=0.22, ear_consecutive_frames=40)
)
```

or from the command line (see CLI above). Key knobs:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ear_threshold` | `0.25` | EAR below which an eye counts as closed |
| `ear_consecutive_frames` | `30` | Frames of closure before DROWSY (~1 s) |
| `mar_threshold` | `0.5` | MAR above which the mouth counts open |
| `mar_consecutive_frames` | `30` | Frames of open mouth before YAWN (~1 s) |
| `yawn_duration_seconds` | `1.0` | Seconds of open mouth before YAWN — FPS-independent fallback |
| `min_blink_frames` | `2` | Minimum closure streak to count a blink |
| `recovery_frames` | `10` | Sustained open eyes needed to leave DROWSY |
| `perclos_window_seconds` | `60.0` | Seconds in the rolling PERCLOS window |
| `perclos_ear_threshold` | `None` | PERCLOS closure threshold (`None` = `ear_threshold`) |
| `perclos_warning_threshold` | `40.0` | PERCLOS % that draws an on-screen warning (no audio) |
| `perclos_drowsy_threshold` | `60.0` | Sustained PERCLOS % that may fire the alarm (calibrate!) |
| `perclos_drowsy_seconds` | `15.0` | How long PERCLOS must stay above the drowsy threshold |
| `head_pose_enabled` | `True` | Master switch for head-pose estimation |
| `head_pitch_down_threshold` | `20.0` | Pitch (+ = down) above which the head counts as nodding down |
| `head_pitch_up_threshold` | `20.0` | Pitch below -this counts as tipped back |
| `head_yaw_left_threshold` | `25.0` | Yaw (+ = driver's left) above which a left turn registers |
| `head_yaw_right_threshold` | `25.0` | Yaw below -this registers as a right turn |
| `head_pose_duration_seconds` | `5.0` | Abnormal pose must be held this long before the reminder |
| `head_pose_smoothing_alpha` | `0.4` | EMA factor for the raw angles (1 = no smoothing) |
| `alert_cooldown_seconds` | `5.0` | Min gap between separate alarm triggers |
| `yawn_siren_seconds` | `2.0` | How long the yawn alarm siren stays on for one confirmed yawn |

### Calibration procedure (recommended per driver)

1. Run `python app.py --no-audio` and look straight at the camera.
2. Read your **live EAR** (up left) — typical value: `0.25 – 0.35`.
3. Close your eyes fully; read the EAR floor — typical: `0.05 – 0.15`.
4. Set `ear_threshold` **between those two values** (midpoint is a good start):
   `python app.py --ear-threshold 0.20`.
5. Repeat for the mouth (`MAR` closed vs. wide yawn) if yawn alerts feel off,
   then adjust `--mar-frames` to your talking habits.

---

## Machine learning extension

The CV rules are reliable and independent; the ML layer is an *optional* module
that learns to separate **ALERT** from **DROWSY** from the features the pipeline
already computes every frame. The ML verdict is a *second opinion*: a display +
logging-only signal that never changes the state machine and never fires the
siren by itself. If no trained model is present the system simply runs without
ML. See [docs/machine_learning.md](docs/machine_learning.md) for the full
workflow.

**Features** — one ordered schema (`src/ml/schema.py`) shared by collection,
training and inference, so the model always sees exactly the features it was
trained on:

```
ear_left, ear_right, ear_mean, mar, perclos,
eye_closed_duration, yawn_duration, pitch, yaw, roll
```

**Workflow**

Collect data across **multiple separate sessions** — never one long recording,
and never a single actor. Each run gets a fresh unique `session_id`, so a
session-grouped train/test split stays leakage-free:

```bash
# 1. Install ML dependencies once
pip install -r requirements-ml.txt

# 2. Collect labelled sessions, ALTERNATING the label (say,n 3 sessions each)
python scripts/collect_ml_data.py --label alert  --samples 300   # Alert session 1
python scripts/collect_ml_data.py --label drowsy --samples 300   # Drowsy session 1
python scripts/collect_ml_data.py --label alert  --samples 300   # Alert session 2
python scripts/collect_ml_data.py --label drowsy --samples 300   # Drowsy session 2
# ... repeat, ideally with different participants (with consent)

# 3. Inspect the collected data (samples, sessions, missing values, balance)
python scripts/dataset_summary.py

# 4. Train + evaluate (Logistic Regression, Random Forest, SVM), save the
#    best model. Session-grouped split prevents data leakage.
python ml/train.py --force --out "data/models/drowsiness_real_{model}.joblib"

# 5. Run with the classifier fusing into the pipeline
python app.py --ml --ml-model data/models/drowsiness_real_random_forest.joblib
```

The trainer prints an honest per-model comparison (accuracy, per-class
precision/recall/F1, macro/weighted averages, `results/model_comparison.csv`,
confusion matrices under `results/`, ranked Random Forest feature-importance)
evaluated on *sessions the model never saw*, and reports exactly how many rows
were dropped for invalid/missing values before training. Class imbalance is
handled with `class_weight="balanced"` (no row duplication). The saved bundle
embeds the feature schema, session split and metrics, so a future mismatch is
detected instead of silently producing wrong predictions.

> **Current status (preliminary):** real data now exists —
> **1,107 samples, 8 sessions** (ALERT 358, DROWSY 749). Trained on a
> session-grouped split (GroupShuffleSplit) with whole held-out sessions:
> **Random Forest** was selected (best DROWSY recall **0.923**, DROWSY F1
> **0.958**, macro F1 0.786) over Logistic Regression (DROWSY recall 0.530)
> and SVM (0.620); saved as `data/models/drowsiness_real_random_forest.joblib`.
> This is a **preliminary single-participant dataset** — the alert test fold is
> a single 20-sample session and the numbers must **not** be treated as
> production-level. `data/models/drowsiness_synthetic_demo.joblib` remains a
> synthetic demo and is not the project's ML accuracy.

The legacy rolling-window extractor (`src/ml/features.py`) and
`scripts/record_session.py` / `scripts/train_classifier.py` remain available
but are superseded by the schema-based pipeline above.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite exercises the mathematical core (EAR/MAR on synthetic landmarks),
the temporal state machine (blink vs. drowsy boundaries, yawn counting,
hysteresis, PERCLOS integration), the rolling-window PERCLOS calculator (0%/50%/
100%, window expiry, invalid/no-face/variable-FPS handling), the head-pose Euler
decomposition / direction classifier / temporal tracker and a synthetic
`solvePnP` round-trip, config overrides and the ML feature extractor — all
**without** a webcam (the `solvePnP` test needs OpenCV, which is already a
runtime requirement).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "Could not open video source" from `--source 1` | Camera index differs; try `0`, `2`, ... or pass a video file path. |
| Low FPS on the preview | Lower resolution (`--width 480 --height 360`) or cap FPS (`--fps 20`). |
| Alerts too sensitive / not sensitive enough | Calibrate EAR/MAR thresholds (see above). |
| `--detector tasks` can't find the model | On mediapipe >= 1.0 it downloads automatically; if downloads are blocked, grab `face_landmarker.task` from the MediaPipe model zoo and drop it in `assets/`. |
| No sound on Linux | `sudo apt install libsdl2-mixer-2.0-0` (pygame). On Windows `winsound` is the fallback. |
| MediaPipe deprecation warning | Expected with newer versions; the `tasks` backend is the future-proof replacement. |

---

## Project structure

```
.
├── app.py                 # ★ user-facing entry point: python app.py
├── assets/                # generated alert siren, optional .task model
├── data/
│   ├── features/           # ML dataset CSVs written by collect_ml_data.py
│   ├── sessions/           # recorded CSV sessions
│   └── models/             # trained classifier bundles (drowsiness_real_*.joblib)
├── docs/
│   ├── DESIGN.md           # deep-dive: architecture, math, ML, extensions
│   └── machine_learning.md # ML: collect -> train -> evaluate -> run
├── ml/
│   └── train.py            # ML training + evaluation + model export
├── scripts/
│   ├── collect_ml_data.py  # ML dataset collection (--label alert|drowsy)
│   ├── dataset_summary.py  # inspect dataset: samples, sessions, missing values
│   ├── record_session.py    # legacy labelled sessions (superseded)
│   ├── train_classifier.py  # legacy ML training (superseded)
│   └── generate_alert_sound.py
├── src/
│   ├── config/settings.py   # every tunable value
│   ├── camera.py
│   ├── pipeline.py          # end-to-end orchestration
│   ├── main.py              # CLI parsing + session loop (shared with app.py)
│   ├── vision/              # detector, landmarks, headpose, renderer
│   ├── dynamics/            # metrics, perclos, headpose, state machine
│   ├── alerts/              # audio, visual, manager
│   ├── ml/                  # schema, dataset, classifier, features (legacy)
│   └── utils/               # logging, fps, session logger
├── tests/                   # pytest suite
├── requirements*.txt
└── pyproject.toml
```

---

## Roadmap / possible extensions

* Per-driver head-pose calibration profile (mount angle + focal-length estimate).
* LSTM/GRU sequence models over the EAR/MAR history instead of the Random Forest.
* Multi-driver sessions (face registration → per-person thresholds).
* Mobile/edge port (TensorFlow Lite / MediaPipe Tasks on Android).
* Video-file output (`--record output.mp4`) for presentations.
* On-device text-to-speech announcements ("Please pull over and rest.").

---

## Disclaimer

This software is a research/demonstration aid only. It is **not** a certified
safety device and must not be relied upon to prevent accidents. Do not use it
while actually driving; always keep control of the vehicle, take breaks, and
follow traffic laws.
