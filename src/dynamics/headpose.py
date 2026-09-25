"""Head-pose estimates: Euler angles, direction and temporal tracking.

Head pose (pitch / yaw / roll) is an *additional* drowsiness signal. It never
fires an alarm on its own: the module below only turns a stream of raw angles
into a smoothed direction plus a ``sustained`` flag that cheap OCR-free rules
can display or log. Like the rest of ``src/dynamics`` this file is
**OpenCV/MediaPipe-free** (pure numpy + math) so it is fully unit-testable;
the OpenCV ``solvePnP`` step that produces the angles lives in
``src.vision.headpose``.

Angle conventions (see docs/DESIGN.md for the derivation)
---------------------------------------------------------
* ``pitch > 0``  -> head **nodding DOWN** (chin tucked toward the chest).
* ``yaw   > 0``  -> head turned toward the **driver's own LEFT**.
* ``roll  > 0``  -> head tilted toward the **driver's own LEFT** shoulder.

These match the decomposition used by ``rotation_matrix_to_euler`` and are
pinned down by synthetic round-trip tests (``tests/test_headpose.py``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HeadPose:
    """One raw head-pose sample in degrees (no temporal state)."""

    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


@dataclass
class TrackedHeadPose:
    """Smoothed angles plus the temporal direction/duration state.

    ``direction`` is ``"NEUTRAL"`` (or ``None`` when nothing has been measured)
    or a dash-joined label such as ``"DOWN"`` or ``"DOWN-LEFT"``. ``sustained``
    becomes True once an abnormal direction has persisted for
    ``duration_threshold_seconds``.
    """

    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None
    direction: str | None = None
    duration: float = 0.0
    sustained: bool = False


def rotation_matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """Decompose a 3x3 rotation matrix into (pitch, yaw, roll) degrees.

    The decomposition is pinned by synthetic tests:

    * a pure ``Rx(+angle)`` (nod) comes back in the *pitch* slot,
    * a pure ``Ry(+angle)`` (turn) comes back in the *yaw* slot,
    * a pure ``Rz(+angle)`` (tilt) comes back in the *roll* slot.

    Moderate driver poses (|angle| < ~45 deg) are recovered accurately; the
    usual small gimbal-lock degradation applies only at extreme angles.
    """
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        # Gimbal lock: pitch == +/-90 deg, only yaw+roll are distinguishable.
        pitch = math.atan2(-r[2, 0], sy)
        yaw = 0.0
        roll = math.atan2(-r[1, 2], r[1, 1])
    else:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def classify_direction(
    pitch: float,
    yaw: float,
    pitch_down_threshold: float,
    pitch_up_threshold: float,
    yaw_left_threshold: float,
    yaw_right_threshold: float,
) -> str:
    """Turn angles into a direction label, or ``"NEUTRAL"`` when within bounds.

    Returns a dash-joined label when the head deviates on more than one axis
    at once (e.g. ``"DOWN-LEFT"``).
    """
    parts: list[str] = []
    if pitch >= pitch_down_threshold:
        parts.append("DOWN")
    elif pitch <= -pitch_up_threshold:
        parts.append("UP")
    if yaw >= yaw_left_threshold:
        parts.append("LEFT")
    elif yaw <= -yaw_right_threshold:
        parts.append("RIGHT")
    return "-".join(parts) if parts else "NEUTRAL"


class HeadPoseTracker:
    """Smooths raw angles, classifies a direction and tracks its duration.

    Mirrors ``PerclosCalculator``'s design: timestamps are real wall-clock
    seconds, so the durations are FPS-independent. A gap longer than
    ``max_gap_seconds`` (face lost, camera covered, dropped frames) restarts
    the current episode instead of silently stretching it.
    """

    def __init__(
        self,
        pitch_down_threshold: float,
        pitch_up_threshold: float,
        yaw_left_threshold: float,
        yaw_right_threshold: float,
        duration_seconds: float,
        smoothing_alpha: float = 0.4,
        max_gap_seconds: float = 1.0,
    ) -> None:
        self.pitch_down_threshold = max(0.0, float(pitch_down_threshold))
        self.pitch_up_threshold = max(0.0, float(pitch_up_threshold))
        self.yaw_left_threshold = max(0.0, float(yaw_left_threshold))
        self.yaw_right_threshold = max(0.0, float(yaw_right_threshold))
        self.duration_seconds = max(0.0, float(duration_seconds))
        self.smoothing_alpha = min(max(float(smoothing_alpha), 0.0), 1.0)
        self.max_gap_seconds = max(float(max_gap_seconds), 1e-3)

        self._smoothed: tuple[float, float, float] | None = None
        self._last_timestamp: float | None = None
        self._episode_since: float | None = None
        self._warned = False

    # -------------------------------------------------------------- public --
    def feed(
        self,
        pose: HeadPose | tuple[float, float, float] | None,
        timestamp: float,
    ) -> TrackedHeadPose:
        """Advance one frame with the latest raw angles.

        ``pose=None`` (or a non-finite angle) keeps the clock but resets the
        smoothed state, so a lost face never leaves a stale direction.
        """
        if pose is not None and isinstance(pose, HeadPose):
            raw = (pose.pitch, pose.yaw, pose.roll)
        elif isinstance(pose, tuple):
            raw = pose
        else:
            raw = None

        if raw is not None and all(math.isfinite(v) for v in raw):
            return self._feed_valid(raw, max(self._last_timestamp or 0.0, timestamp))
        return self._feed_invalid(timestamp)

    def reset(self) -> None:
        """Forget the smoothing state and any in-progress episode."""
        self._smoothed = None
        self._last_timestamp = None
        self._episode_since = None
        self._warned = False

    # ------------------------------------------------------------ internals --
    def _feed_valid(
        self, raw: tuple[float, float, float], timestamp: float
    ) -> TrackedHeadPose:
        # A long gap since the *last sample* (face lost, camera covered,
        # dropped frames) discards the older angles and starts a fresh episode:
        # the unobserved span must not count as "head held still".
        gap = (
            self._last_timestamp is not None
            and timestamp - self._last_timestamp > self.max_gap_seconds
        )
        if gap:
            self._smoothed = None
            self._episode_since = None
        self._last_timestamp = timestamp

        if self._smoothed is None:
            smoothed = tuple(raw)
        else:
            a = self.smoothing_alpha
            smoothed = tuple(
                a * v + (1.0 - a) * prev for v, prev in zip(raw, self._smoothed)
            )
        self._smoothed = smoothed

        pitch, yaw, roll = smoothed
        direction = classify_direction(
            pitch,
            yaw,
            self.pitch_down_threshold,
            self.pitch_up_threshold,
            self.yaw_left_threshold,
            self.yaw_right_threshold,
        )

        duration = 0.0
        sustained = False
        if direction == "NEUTRAL":
            self._episode_since = None
        else:
            if self._episode_since is None:
                self._episode_since = timestamp
            duration = max(0.0, timestamp - self._episode_since)
            sustained = duration >= self.duration_seconds

        if sustained and not self._warned:
            logger.warning(
                "Sustained abnormal head pose: %s held for %.1fs "
                "(pitch=%.1f yaw=%.1f roll=%.1f)",
                direction, duration, pitch, yaw, roll,
            )
            self._warned = True
        elif direction == "NEUTRAL":
            self._warned = False

        return TrackedHeadPose(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            direction=direction,
            duration=duration,
            sustained=sustained,
        )

    def _feed_invalid(self, timestamp: float) -> TrackedHeadPose:
        self._last_timestamp = timestamp
        self._smoothed = None
        self._episode_since = None
        self._warned = False
        return TrackedHeadPose(direction=None)


__all__ = [
    "HeadPose",
    "TrackedHeadPose",
    "rotation_matrix_to_euler",
    "classify_direction",
    "HeadPoseTracker",
]
