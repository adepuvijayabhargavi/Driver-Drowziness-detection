"""Application configuration.

Every tunable behaviour of the system lives here. Thresholds are *configurable*
on purpose: the same numbers do not work for everyone or every camera, so the
CLI (``--ear-threshold``, ``--mar-threshold``, ...) lets you override any value
at runtime without touching the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# Absolute location of the project root (``.../Driver Drowsiness Detection and Alert System``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ASSETS_DIR = Path(os.environ.get("DROWSINESS_ASSETS", PROJECT_ROOT / "assets"))
DATA_DIR = Path(os.environ.get("DROWSINESS_DATA", PROJECT_ROOT / "data"))
SESSIONS_DIR = DATA_DIR / "sessions"
MODELS_DIR = DATA_DIR / "models"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_ALERT_SOUND_PATH = ASSETS_DIR / "alert_beep.wav"


@dataclass(frozen=True)
class CameraConfig:
    """Input source settings."""

    source: int | str = 0  # 0 -> first webcam; or a path to a video file.
    width: int = 640       # Requested capture width (the driver may override).
    height: int = 480
    target_fps: float = 30.0  # Upper bound; never exceeds the camera's real fps.


@dataclass(frozen=True)
class DetectionSettings:
    """Geometry, thresholds and temporal windows for drowsiness rules."""

    # ------------------------------------------------------------------ EAR ---
    # Eye Aspect Ratio below which the eye is considered closed.
    # Typical alert eyes sit around 0.28-0.35; closed eyes fall below ~0.15.
    # Needs calibration per driver / camera / glasses.
    ear_threshold: float = 0.25
    # Consecutive frames with EAR < threshold before the driver is DROWSY.
    # At 30 fps, 30 frames == 1 full second of closed eyes.
    ear_consecutive_frames: int = 30

    # ------------------------------------------------------------------ MAR ---
    # Mouth Aspect Ratio above which the mouth is considered open (yawn).
    # A relaxed mouth is very small (MAR ~ 0.05-0.2). A full yawn typically
    # drives MAR to 0.5-1.2+, but face shape and mouth width vary a lot, so 0.5
    # is the default -- the old 0.55 was too high for many people and yawns with
    # a wide mouth corner spread never crossed it. Tune per driver with
    # ``--mar-threshold``.
    mar_threshold: float = 0.5
    # Consecutive frames with MAR > threshold to register a yawn. A yawn is also
    # confirmed by ``yawn_duration_seconds`` below -- whichever bound is reached
    # first wins, so detection no longer depends on the camera FPS.
    mar_consecutive_frames: int = 30
    # Seconds the mouth must stay above the MAR threshold to register a yawn.
    # At 30 FPS this matches mar_consecutive_frames (1 s); at lower FPS it
    # guarantees a real yawn is still caught instead of silently requiring a
    # longer streak than the person can physically sustain.
    yawn_duration_seconds: float = 1.0

    # ----------------------------------------------------------------- eyes ---
    # Minimum closure-streak (frames) that counts as a blink (noise filter).
    min_blink_frames: int = 2
    # After a DROWSY alert, this many consecutive "open eye" frames are needed
    # before the driver returns to ALERT (hysteresis prevents flicker).
    recovery_frames: int = 10

    # ---------------------------------------------------------------- misc ---
    # Treat a "no face" run this many frames long as a visible warning.
    no_face_warning_frames: int = 60

    # -------------------------------------------------------------- PERCLOS ---
    # Percentage of Eye Closure: the fraction of the last
    # ``perclos_window_seconds`` of *real time* in which EAR <
    # ``perclos_ear_threshold`` (measured with timestamps, not a fixed frame
    # count). PERCLOS is an ADDITIONAL drowsiness signal -- it augments the EAR
    # streak alarm and never replaces it.
    perclos_enabled: bool = True
    perclos_window_seconds: float = 60.0
    # EAR below which a frame counts as "eyes substantially closed" for the
    # PERCLOS statistic. ``None`` = follow ``ear_threshold`` so both signals
    # share the same definition of "closed".
    perclos_ear_threshold: float | None = None
    # PERCLOS >= this percent draws a persistent on-screen warning (visual
    # only, never audio). Normal alert blinking accounts for roughly 5-15 % of
    # a minute, so 40 % is already a strong warning signal.
    perclos_warning_threshold: float = 40.0
    # PERCLOS sustained >= this percent for ``perclos_drowsy_seconds`` can push
    # the state machine into DROWSY (same siren as the EAR alarm), subject to
    # the usual alert cooldown. The defaults are deliberately conservative and
    # MUST be validated/calibrated per driver -- they are not a certified
    # fatigue measurement.
    perclos_drowsy_threshold: float = 60.0
    perclos_drowsy_seconds: float = 15.0

    # ------------------------------------------------------------- HEAD POSE ---
    # Head pitch/yaw/roll estimated from the face geometry with solvePnP.
    # Head pose is an ADDITIONAL *display-only* signal: it NEVER fires an alarm
    # and never pushes the state machine into DROWSY -- it only draws an
    # on-screen reminder once a deviation is held for ``head_pose_duration_seconds``,
    # to help catch the classic "head falling forward" sleepy posture.
    # Angle conventions (see docs/DESIGN.md): pitch > 0 = nodding DOWN,
    # yaw > 0 = turned to the driver's own LEFT, roll > 0 = tilted toward the
    # driver's own LEFT shoulder. The absolute values are approximate (single
    # uncalibrated camera); the thresholds below pick the *signals*, not the
    # exact degrees.
    head_pose_enabled: bool = True
    # |yaw| threshold over which a turn registers as LEFT (yaw > 0) or RIGHT.
    head_yaw_right_threshold: float = 25.0
    head_yaw_left_threshold: float = 25.0
    # Pitch threshold: > 0 means pitching down (chin on chest / watching a
    # phone), < 0 means tipping the head back.
    head_pitch_down_threshold: float = 20.0
    head_pitch_up_threshold: float = 20.0
    # Seconds an abnormal posture must be held before the on-screen "HEAD DOWN"
    # reminder appears. Deliberately conservative: a quick glance at the mirror
    # or dashboard must never alarm.
    head_pose_duration_seconds: float = 5.0
    # Exponential moving-average factor for the raw angles (1.0 = no smoothing,
    # 0.0 = never move). 0.4 keeps the latency below ~0.1 s at 30 fps.
    head_pose_smoothing_alpha: float = 0.4


@dataclass(frozen=True)
class AlertSettings:
    """Behaviour of the visual + audio warning system."""

    # Cooldown between *separate* alarm triggers (seconds). Prevents a single
    # drowsy spell from spamming the siren with dozens of re-triggers. The very
    # first yawn alert is never blocked by the cooldown.
    alert_cooldown_seconds: float = 5.0
    # How long the loud siren stays on for one confirmed yawn. Yawns get the
    # same audio call as the drowsiness alarm (AlertManager -> AudioAlert),
    # but are auto-stopped after this window so the alarm cannot loop forever.
    yawn_siren_seconds: float = 2.0
    # Loudness (0.0-1.0) for the generated siren.
    audio_volume: float = 0.9
    # Fires the siren on sustained yawning (same alert as drowsiness).
    yawn_alert_enabled: bool = True
    # Master switch for audio (e.g. run with --no-audio).
    audio_enabled: bool = True


@dataclass(frozen=True)
class MLConfig:
    """Optional machine-learning classifier that fuses with the CV rules.

    The rules themselves are reliable and never depend on this module; the ML
    classifier is an *add-on* that can raise confidence / catch gradual
    deterioration. If no trained model is found the system silently run on the
    classic CV rules only.
    """

    enabled: bool = False
    model_path: Path | None = None  # Default: MODELS_DIR / "drowsiness_real_random_forest.joblib"
    # Probability above which the ML model alone can flag drowsiness.
    confidence_threshold: float = 0.85
    feature_window: int = 45  # Kept for legacy FeatureExtractor compatibility.


@dataclass(frozen=True)
class DLConfig:
    """Optional temporal deep-learning classifier (LSTM over frame sequences).

    Like the ML layer, the DL verdict is a *display + log-only* second opinion:
    it never changes the state machine and never fires the siren by itself --
    the :class:`~src.alerts.manager.AlertManager` stays the single authority
    for audio/visual alerts. When the model or its scaler is missing the
    pipeline falls back cleanly to the CV rules (+ optional ML).
    """

    enabled: bool = False
    model_path: Path | None = None  # Defaults to MODELS_DIR / "drowsiness_lstm.keras"
    scaler_path: Path | None = None  # Defaults to MODELS_DIR / "drowsiness_lstm_scaler.joblib"
    # Frames per temporal window. None = auto-detect the window the trained
    # LSTM expects (loaded from its metadata / input shape).
    sequence_length: int | None = None
    # Probability above which a DROWSY vote is worth logging.
    confidence_threshold: float = 0.85


@dataclass(frozen=True)
class Settings:
    """Top-level configuration aggregate."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    alert: AlertSettings = field(default_factory=AlertSettings)
    ml: MLConfig = field(default_factory=MLConfig)
    dl: DLConfig = field(default_factory=DLConfig)

    # Runtime behavior (kept here to be overridable from the CLI as well).
    show_window: bool = True
    save_log: bool = False
    log_path: Path | None = None  # Defaults to SESSIONS_DIR / session timestamped.csv
    log_level: str = "INFO"

    def with_detection(self, **changes) -> "Settings":
        """Return a copy with the given detection settings replaced."""
        return replace(self, detection=replace(self.detection, **changes))

    def with_camera(self, **changes) -> "Settings":
        return replace(self, camera=replace(self.camera, **changes))

    def with_alert(self, **changes) -> "Settings":
        return replace(self, alert=replace(self.alert, **changes))

    def with_ml(self, **changes) -> "Settings":
        return replace(self, ml=replace(self.ml, **changes))

    def with_dl(self, **changes) -> "Settings":
        return replace(self, dl=replace(self.dl, **changes))

    def with_runtime(self, **changes) -> "Settings":
        return replace(self, **changes)


# The canonical default settings used by every entry point that does not
# receive an explicit Settings object.
DEFAULT_SETTINGS = Settings()

__all__ = [
    "PROJECT_ROOT",
    "ASSETS_DIR",
    "DATA_DIR",
    "SESSIONS_DIR",
    "MODELS_DIR",
    "FEATURES_DIR",
    "RESULTS_DIR",
    "DEFAULT_ALERT_SOUND_PATH",
    "CameraConfig",
    "DetectionSettings",
    "AlertSettings",
    "MLConfig",
    "DLConfig",
    "Settings",
    "DEFAULT_SETTINGS",
]
