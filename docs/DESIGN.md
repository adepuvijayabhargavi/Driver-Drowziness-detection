# Design notes

Deep-dive into the architecture, mathematics and design decisions of the driver
drowsiness detection system. The README is the user manual; this document is
about *why* things are built this way.

---

## 1. Why an aspect-ratio approach?

Drowsiness has several measurable physiological correlates:

1. **Micro-sleeps** — brief, involuntary eye closures (300 ms – 3 s).
2. **Slow eyelid closure** — heavy eyelids drift down gradually (PERCLOS-like).
3. **Yawning** — wide, sustained mouth opening.
4. **Head nodding** — sudden head drops (3-D head pose via `solvePnP`;
   implemented as a *display-only* supporting signal — see § 4c).
5. **Blink pattern changes** — increased blink frequency or long blinks.

The EAR/MAR approach was chosen because MediaPipe gives us *direct, dense*
landmark geometry at low cost, from which the eye/mouth openness is trivially
and *robustly* measurable. It is far less sensitive to camera angle, lighting
and skin tone than, say, a raw CNN eye state classifier, and it runs on CPU.

EAR/MAR are **scale- and rotation-invariant**: aspect ratios cancel both the
absolute size of the face (distance to camera) and in-plane rotation. This is
exactly why ratios, not pixel distances, are used.

## 2. System architecture

The code is split into layers with a strict inward dependency rule:

```
           main.py  (CLI, I/O, window loop)
              │
         pipeline.py  (orchestration, no external side-effects)
              │
   ┌──────────┼──────────────────────┐
   │          │                      │
 vision/   dynamics/             alerts/           ml/      utils/
 (cv)     (pure math)         (pygame/cv)       (sklearn)
```

* **`vision`** — talks to MediaPipe/OpenCV. Returns neutral `FaceData`.
* **`dynamics`** — 100% dependency-free math + state machine. *No* OpenCV, *no*
  MediaPipe imports. This is what makes tests trivial and reliable.
* **`alerts`** — side effects only (sound, screen). Never affects the metric
  computation.
* **`ml`** — optional. Depends on dynamics features only.
* **`pipeline`** — the only place that composes everything per frame, plus the
  visual annotation of the pass-through frame.

**Design decoupling decision**: lazy imports. `metrics.py`, `state_machine.py`
and `features.py` import *nothing* from `cv2`/`mediapipe`. `detector.py`,
`renderer.py` and `camera.py` import them lazily inside methods. The result:
the full test suite runs with NumPy only and the project compiles on machines
without OpenCV.

## 3. The state machine

```
for each frame:
    if face lost:  push nothing, hold counters, state -> UNKNOWN
    else:
        if EAR < ear_threshold:   eye_closed_frames += 1
        else:                     close-open transition: blink scored?
                                  (streak in [min_blink_frames, ear_consecutive_frames))
                                  eye_closed_frames = 0

        if MAR > mar_threshold:   mouth_open_frames += 1
        else:                     close-open transition: yawn scored?
                                  mouth_open_frames = 0

        state = DROWSY  if eye_closed_frames >= ear_consecutive_frames
                YAWNING if mouth_open_frames >= mar_consecutive_frames
                (recovery handled with hysteresis counters)
```

**Hysteresis** is the trick that keeps alarms single-shot and unobtrusive:
once `DROWSY`, the machine only returns to `ALERT` after `recovery_frames`
(e.g. 10) consecutive open-eye frames. Combined with the alert cooldown
(`alert_cooldown_seconds`), a marginally-drowsy driver gets exactly one clean
alarm rather than a ratchet of warnings.

**Blink rate** is maintained as a rolling window: one sample per frame, `1.0`
set on the frame a blink completes, `0.0` otherwise. The `blinks / minute`
figure is therefore an EMA-like rate over the last 60 s and a strong
statistical signal for an ML extension.

## 4. EAR & MAR derivations

**EAR** (Soukupová & Čech, 2016):

```
EAR = (d(p2,p6) + d(p3,p5)) / (2·d(p1,p4))
```

with `d` Euclidean distance. For the MediaPipe model the six points are:

| | p1 | p2 | p3 | p4 | p5 | p6 |
|---|---|---|---|---|---|---|
| RIGHT eye | 33 | 160 | 158 | 133 | 153 | 144 |
| LEFT eye | 362 | 385 | 387 | 263 | 373 | 380 |

**MAR**:

```
MAR = d(upper inner lip, lower inner lip) / d(left corner, right corner)
   = d(13, 14) / d(61, 291)
```

Both ratios are computed from *normalized* MediaPipe coordinates, i.e. we keep
everything in [0, 1] face space and only scale to pixel space for drawing.

## 4b. PERCLOS — the cumulative eye-closure signal

`src/dynamics/perclos.py` implements **PERCLOS** (Percentage of Eye Closure):
the share of *real time*, over a sliding window, during which `EAR` stays below
`perclos_ear_threshold` (default: the same `ear_threshold` used by the EAR
alarm).

```
PERCLOS = (closed-eye time in window) / (valid monitoring time in window) × 100
```

Design decisions:

* **Timestamp-based, not frame-count based.** Every valid sample opens an
  interval that the *next* valid sample closes; intervals are measured from the
  real timestamps, so a variable-FPS or dropped-frame camera does not distort
  the ratio (a naive `closed/total frames` would).
* **Gap handling.** An interval longer than `max_gap_seconds` (1 s) is
  discarded — the seconds the driver looked away or the camera was covered are
  counted as *neither* open nor closed time.
* **Noise-free accounting.** Missing / `NaN` / `inf` / negative EAR values are
  ignored, so an invalid measurement can never inflate closed-eye time.
* **O(1)-amortized rolling window.** Finished intervals live in a `deque` and
  are pruned as they age out of `window_seconds`.
* **Additive, conservative integration.** PERCLOS *adds to* the DROWSY decision:
  only a value sustained above `perclos_drowsy_threshold` for
  `perclos_drowsy_seconds` can enter DROWSY (reusing the existing siren/cooldown
  path). The EAR-streak alarm is byte-for-byte unchanged; a single-frame PERCLOS
  change can never alarm.
* **Calibration.** Defaults (60 s window, 40 % visual warning, 60 % drowsy ×
  15 s) sit well above normal blink levels (~5–15 % of a minute). These must be
  validated per driver and camera; PERCLOS is one input to a multi-signal
  decision, never proof of fatigue on its own.

## 4c. Head pose — Euler angles behind a "sleepy posture" display

Head pose is the **fourth** physiological correlate (§ 1.4). Its architecture
is split exactly along the dependency rule: `src/vision/headpose.py` produces
raw *angles* with OpenCV, and `src/dynamics/headpose.py` turns them into a
*direction + duration* (pure Python, no OpenCV).

**Fitting.** Six MediaPipe landmarks (nose tip `1`, chin `152`, the two eye
corners `33`/`263`, the two mouth corners `291`/`61`) are fitted to a canonical
rigid 3-D face model (millimetres, origin at the nose tip, +x toward the
*subject's left* so a frontal face aligns with the camera) using
`cv2.solvePnP` (ITERATIVE). The recovered object→camera rotation matrix is
decomposed with `rotation_matrix_to_euler`, whose axis/sign convention is
pinned by synthetic round-trip tests:

```
pitch (about camera X)  : + = nodding DOWN  (chin toward chest)  — sleep cue
yaw   (about camera Y)  : + = driver's own LEFT,  − = RIGHT
roll  (about camera Z)  : + = tilt toward driver's own LEFT
```

The camera matrix uses the standard single-camera approximation
`focal ≈ frame width`, `principal point = frame centre`, zero distortion.

**Why display-only and never an alarm.** A driver turns their head for *benign*
reasons constantly — rear mirror (up to ~40° yaw), side mirrors, speedometer,
a phone. The earlier failure modes in the eye/mouth domain (blink = almost
sleep) have their head-pose equivalent: a 5-second glance at the mirror is
**not** fatigue. Alarming on head pose alone would be the single most
false-positive-prone rule in the system. Instead:

* raw angles → EMA (`head_pose_smoothing_alpha = 0.4`) to kill landmark jitter;
* `classify_direction` turns the smoothed angles into `DOWN / UP / LEFT /
  RIGHT / DOWN-LEFT ...` (thresholds `head_pitch_down_threshold` etc.);
* `HeadPoseTracker` accumulates *held* time while a direction is abnormal
  (mirroring `PerclosCalculator`'s timestamp/gap semantics: a feed gap > 1 s
  — face lost, camera covered — restarts the episode, so an unobservable
  driver is never counted as "head held still");
* `head_pose_sustained` (abnormal for ≥ `head_pose_duration_seconds`) only
  draws a neutral-yellow on-screen reminder — no audio, no state change.

**Calibration / accuracy caveats.** A single uncalibrated webcam cannot
produce exact absolute degrees: the focal-length shortcut and the generic face
model bias the *magnitude*. The *signs* and *relative* magnitudes are reliable
(which is why the thresholds are best read as "how far the driver must deviate
to register"), and the synthetic round-trip tests in `tests/test_headpose.py`
verify the conventions for every axis and both directions. Per-driver tuning
(`--head-pitch-down`, `--head-yaw-left`, ...) and camera-mount mounting are
expected.

## 5. Why not a deep neural network for everything?

The classic CV approach is *interpretable, deterministic and debuggable* — you
can print the exact reason DROWSY fired and re-run it frame by frame. An
end-to-end DNN classifier on RGB frames would require:

* a large, annotated drowsiness dataset (rare, privacy-sensitive),
* careful train/test splits across identities,
* GPU training and heavy runtime cost.

The system therefore uses **Deep Learning exactly where it is best**: MediaPipe
FaceMesh is a DNN that extracts facial geometry reliably. The classification on
top is a transparent rule-engine plus an optional classical ML model. This
hybrid is the industry standard for this problem.

## 6. The ML extension in more detail

**Training data**: `scripts/record_session.py` streams EAR/MAR + live state to a
CSV and allows manual label correction (`a`/`d`/`y`/`u`). Because labels are
per-session and sequential, rows inherit the most recent label.

**Feature vector** (10 dims, trailing window of `W` frames):

| feature | intuition |
|---|---|
| `ear_mean` | overall how open the eyes are |
| `ear_min`, `ear_q10` | depth of micro-closures |
| `ear_std` | erratic/energetic lids vs droop |
| `mar_mean`, `mar_max`, `mar_std` | mouth openness persistence |
| `eye_closed_fraction` | percentage of window with EAR < threshold |
| `mouth_open_fraction` | percentage of window with MAR > threshold |
| `mar_range` | mouth movement dynamics |

**Model**: `RandomForestClassifier` (200 trees, depth 10, `class_weight` =
`"balanced"`) — robust to the highly imbalanced per-frame class distribution,
non-parametric (thresholds of different people differ!), and gives free feature
importances. Reported metrics: accuracy, macro-F1, confusion matrix.

**Fusion at inference**: ML produces `P(drowsy)`. If `P(drowsy) ≥
confidence_threshold` the system flags drowsiness even when the CV window is
not yet exhausted. The CV state machine and alert logic are unchanged.

## 7. Known limitations

* Cannot detect the face in total darkness (no IR camera support out of the box).
* Sunglasses can occlude eye landmarks → EAR unreliable; consider `refine_landmarks`
  and per-driver calibration.
* A driver who looks away ("no face") pauses the counters: the alarm neither
  resets nor trials. Intentional, but a caveat.
* MediaPipe's `solutions` API is deprecated (works today, emits warnings). The
  `tasks` backend is provided as the migration path (`--detector tasks`).

## 8. Extension ideas

* **Head-pose nodding detector**: use the 468-point 3D model + `cv2.solvePnP`
  with a standard 6-point face reference; nod detection is a strong drowsiness
  corroborator (see Radar diagram in reviews).
* **Per-driver calibration store**: persist EAR baseline/profile per face id.
* **Sequence models**: replace/extend the Random Forest with an LSTM over the
  feature windows — capture transitions ("eyes closing over 0.5 s").
* **Robustness**: median-filter EAR/MAR streams; drop occlusion/infrared-camera
  scenarios.
* **Integration**: CAN-bus / OBD-II speed signal to suppress false alarms at
  low speed, plus driver-facing display.