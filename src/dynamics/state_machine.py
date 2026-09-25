"""Temporal drowsiness state machine.

The CV metrics on a *single* frame are noisy and momentary. A half-blink that
lasts 2 frames must never trigger an alarm. This module turns the frame-level
EAR / MAR streams into time-consistent events:

    * DROWSY - eyes below the EAR threshold for ``ear_consecutive_frames``.
    * YAWN   - mouth above the MAR threshold for the FIRST of
               ``mar_consecutive_frames`` (frame count) OR ``yawn_duration_seconds``
               (wall-clock duration at the real FPS). Frame-count alone used to
               make yawning impossible on low-FPS cameras, because a streak that
               is long enough in seconds could never accumulate in frames.
    * BLINK  - a short eye closure that re-opens well short of the drowsy
               window (its frequency is itself a drowsiness signal).
    * recovery - hysteresis: once DROWSY, the driver must show ``recovery_frames``
                 of open eyes before returning to ALERT (kills flicker).

The yawn counter is *separate* from the eye-closed counter: the two are
independent streams, and the yawn counter only resets when MAR falls back below
the threshold.

The class is pure Python so it is fully deterministic and unit-testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from src.config.settings import DetectionSettings

logger = logging.getLogger(__name__)


class DriverState(Enum):
    """Inferred high-level driver state."""

    ALERT = "ALERT"      # eyes open, no prolonged closures
    YAWNING = "YAWNING"  # sustained mouth opening detected
    DROWSY = "DROWSY"    # eyes closed beyond the allowed time window
    UNKNOWN = "UNKNOWN"  # no face / unusable measurements this frame


@dataclass
class FrameReport:
    """Everything inferred from a single processed frame."""

    frame_index: int
    timestamp: float          # seconds (monotonic)
    face_present: bool
    ear: float | None         # average EAR over both eyes, None if no face
    mar: float | None
    state: DriverState
    drowsy: bool              # driver is (still) in the DROWSY state
    yawning: bool             # a sustained mouth opening is in progress
    eye_closed_frames: int    # current uninterrupted closure streak
    mouth_open_frames: int    # current uninterrupted mouth-open streak
    blink_count: int          # total blinks scored this session
    yawn_count: int           # total yawns scored this session
    blink_rate_per_min: float # rolling blinks per minute
    no_face_frames: int       # uninterrupted no-face streak
    triggered: bool = False   # True exactly on the transition into DROWSY
    yawn_triggered: bool = False  # True exactly on the frame a yawn is confirmed
    yawn_frames: int = 0      # current uninterrupted yawn (mouth-open) counter
    yawn_duration: float = 0.0  # seconds the mouth has stayed above the MAR threshold
    perclos: float | None = None   # rolling PERCLOS (%) over the recent window
    perclos_drowsy: bool = False   # PERCLOS sustained above the drowsy threshold
    ear_left: float | None = None   # EAR of the driver's left eye (set by pipeline)
    ear_right: float | None = None  # EAR of the driver's right eye (set by pipeline)
    eye_closed_duration: float = 0.0  # seconds the eyes have stayed closed
    head_pitch: float | None = None   # smoothed head pitch (deg; + = down)
    head_yaw: float | None = None     # smoothed head yaw (deg; + = driver's left)
    head_roll: float | None = None    # smoothed head roll (deg; + = driver's left)
    head_direction: str | None = None  # "DOWN", "LEFT", "DOWN-LEFT", ... or None
    head_pose_duration: float = 0.0  # seconds the current pose has been held
    head_pose_sustained: bool = False  # abnormal pose held >= the configured window
    ml_prediction: str | None = None  # ML verdict: "ALERT" / "DROWSY" (if enabled)
    ml_confidence: float | None = None  # probability of the predicted class
    dl_prediction: str | None = None  # DL (LSTM) verdict: "ALERT" / "DROWSY"
    dl_confidence: float | None = None  # probability of the predicted class
    dl_warming_up: bool = False  # True until the rolling DL window is filled


class _RollingBoolCounter:
    """Rolling window: stores one sample per frame for ``maxsize`` frames."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = max(1, maxsize)
        self._values: list[float] = []

    def push(self, value: float) -> None:
        self._values.append(value)
        if len(self._values) > self.maxsize:
            self._values.pop(0)

    def total(self) -> int:
        return sum(self._values)

    def rate_per_minute(self, fps: float) -> float:
        if fps <= 0 or not self._values:
            return 0.0
        span_seconds = len(self._values) / fps
        if span_seconds <= 0:
            return 0.0
        return (self.total() * 60.0) / span_seconds


class DrowsinessStateMachine:
    """Stateful tracker that converts EAR/MAR streams into drowsiness events."""

    def __init__(
        self,
        settings: DetectionSettings | None = None,
        fps: float = 30.0,
        blink_rate_window_seconds: float = 60.0,
    ) -> None:
        self.settings = settings or DetectionSettings()
        self.fps = max(fps, 1e-3)
        self._blink_window = _RollingBoolCounter(
            maxsize=max(1, int(blink_rate_window_seconds * self.fps))
        )

        self.frame_index: int = 0

        self._eye_closed_frames = 0
        self._eye_closed_since: float | None = None  # monotonic ts eyes first closed
        self.yawn_counter = 0          # separate counter for mouth-open stays
        self._mouth_open_since: float | None = None
        self._no_face_frames = 0

        self.blink_count: int = 0
        self.yawn_count: int = 0
        self._in_yawn = False
        self._recovery_count = 0

        self._perclos_high_since: float | None = None
        self._perclos_active = False

        self.state: DriverState = DriverState.ALERT

    # ------------------------------------------------------------- external --
    def update(
        self,
        ear: float | None,
        mar: float | None,
        timestamp: float = 0.0,
        perclos: float | None = None,
    ) -> FrameReport:
        """Advance one frame with the current (ear, mar); returns the report."""
        self.frame_index += 1
        face_present = ear is not None and mar is not None

        if not face_present:
            self._no_face_frames += 1
            # Freeze all counters: we must not *reward* (reset) or *punish*
            # (increment) drowsiness while the driver is unobservable.
            report = self._build_report(ear, mar, timestamp)
            if self.state is DriverState.ALERT:
                report.state = DriverState.UNKNOWN
            else:
                report.state = self.state  # don't downgrade hidden hazards
            report.drowsy = self.state is DriverState.DROWSY
            report.yawning = self.state is DriverState.YAWNING
            report.no_face_frames = self._no_face_frames
            return report

        self._no_face_frames = 0

        eye_closed = ear < self.settings.ear_threshold
        mouth_open = mar > self.settings.mar_threshold

        # ------------------------------------------------------------ eyes ---
        eye_closed_duration = 0.0
        if eye_closed:
            self._eye_closed_frames += 1
            self._recovery_count = 0
            if self._eye_closed_since is None:
                self._eye_closed_since = timestamp
            eye_closed_duration = max(0.0, timestamp - self._eye_closed_since)
        else:
            # Cleanly finished closure streak: score a blink when it was short.
            if 0 < self._eye_closed_frames < self.settings.ear_consecutive_frames:
                if self._eye_closed_frames >= self.settings.min_blink_frames:
                    self.blink_count += 1
                    self._blink_window.push(1.0)
                else:
                    self._blink_window.push(0.0)
            else:
                self._blink_window.push(0.0)
            self._eye_closed_frames = 0
            self._eye_closed_since = None
            self._recovery_count += 1

        # ----------------------------------------------------------- mouth ---
        if mouth_open:
            if self._mouth_open_since is None:
                self._mouth_open_since = timestamp
                logger.debug(
                    "MAR crossed threshold: MAR=%.3f > %.3f (frame %d)",
                    mar, self.settings.mar_threshold, self.frame_index,
                )
            self.yawn_counter += 1
            logger.debug(
                "Yawn counter increasing: %d/%d frames, held %.2fs (MAR=%.3f)",
                self.yawn_counter,
                self.settings.mar_consecutive_frames,
                max(0.0, timestamp - self._mouth_open_since),
                mar,
            )
        else:
            if self._in_yawn:
                self.yawn_count += 1
            self._in_yawn = False
            self.yawn_counter = 0
            self._mouth_open_since = None

        open_duration = (
            max(0.0, timestamp - self._mouth_open_since)
            if mouth_open and self._mouth_open_since is not None
            else 0.0
        )
        # A yawn is confirmed by the FIRST bound reached: enough consecutive
        # frames OR enough wall-clock time (FPS-independent).
        yawning_now = mouth_open and (
            self.yawn_counter >= self.settings.mar_consecutive_frames
            or open_duration >= self.settings.yawn_duration_seconds
        )
        yawn_just_confirmed = yawning_now and not self._in_yawn
        if yawn_just_confirmed:
            self._in_yawn = True
            logger.warning(
                "Yawn confirmed: MAR=%.3f held for %d frames (%.2fs)",
                mar, self.yawn_counter, open_duration,
            )

        # ------------------------------------------------------------ perclos ---
        # PERCLOS is an ADDITIONAL drowsiness signal: a value sustained above
        # ``perclos_drowsy_threshold`` for ``perclos_drowsy_seconds`` may enter
        # DROWSY, but the temporary blip of a single frame never can. It is
        # purely additive -- the EAR-streak path above is completely unchanged.
        perclos_active = False
        perclos_just_activated = False
        if self.settings.perclos_enabled and perclos is not None:
            if perclos >= self.settings.perclos_drowsy_threshold:
                if self._perclos_high_since is None:
                    self._perclos_high_since = timestamp
                perclos_active = (
                    timestamp - self._perclos_high_since
                ) >= self.settings.perclos_drowsy_seconds
                perclos_just_activated = perclos_active and not self._perclos_active
                self._perclos_active = perclos_active
            else:
                self._perclos_high_since = None
                self._perclos_active = False

        # ----------------------------------------------------------- state ---
        drowsy_now = (
            self._eye_closed_frames >= self.settings.ear_consecutive_frames
        )
        if drowsy_now or perclos_active:
            self.state = DriverState.DROWSY
            self._recovery_count = 0
        elif self.state is DriverState.DROWSY:
            # Hysteresis: a DROWSY driver needs sustained open eyes to recover.
            if self._recovery_count >= self.settings.recovery_frames:
                self.state = DriverState.ALERT
        elif yawning_now:
            self.state = DriverState.YAWNING
        elif self.state is DriverState.YAWNING:
            # The mouth closed again; the alert already fired on the confirm
            # frame via ``yawn_triggered``, so we can leave YAWNING freely.
            self.state = DriverState.ALERT
        # else: ALERT + nothing open -> stays ALERT

        report = self._build_report(ear, mar, timestamp)
        report.state = self.state
        report.drowsy = self.state is DriverState.DROWSY
        report.yawning = yawning_now or self.state is DriverState.YAWNING
        report.yawn_triggered = yawn_just_confirmed
        report.yawn_frames = self.yawn_counter
        report.yawn_duration = open_duration
        report.eye_closed_duration = eye_closed_duration
        report.perclos = perclos
        report.perclos_drowsy = perclos_active
        report.triggered = (
            drowsy_now and self._eye_closed_frames == self.settings.ear_consecutive_frames
        ) or perclos_just_activated
        report.blink_rate_per_min = self._blink_window.rate_per_minute(self.fps)
        return report

    # --------------------------------------------------------- lifecycle ----
    def reset(self) -> None:
        """Reset every counter (used when restarting a session)."""
        self.__init__(self.settings, self.fps, self._blink_window.maxsize)

    # ------------------------------------------------------------ helpers ---
    def _build_report(
        self, ear: float | None, mar: float | None, timestamp: float
    ) -> FrameReport:
        return FrameReport(
            frame_index=self.frame_index,
            timestamp=timestamp,
            face_present=ear is not None and mar is not None,
            ear=ear,
            mar=mar,
            state=self.state,
            drowsy=False,
            yawning=False,
            eye_closed_frames=self._eye_closed_frames,
            mouth_open_frames=self.yawn_counter,
            blink_count=self.blink_count,
            yawn_count=self.yawn_count,
            blink_rate_per_min=0.0,
            no_face_frames=self._no_face_frames,
            triggered=False,
            yawn_triggered=False,
            yawn_frames=self.yawn_counter,
        )

    @property
    def eye_closed_frames(self) -> int:
        return self._eye_closed_frames

    @property
    def mouth_open_frames(self) -> int:
        return self.yawn_counter


__all__ = ["DriverState", "FrameReport", "DrowsinessStateMachine"]
