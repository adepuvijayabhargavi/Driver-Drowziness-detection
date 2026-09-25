"""Tests for head-pose estimation, classification and temporal tracking.

The Euler decomposition and the tracker are pure Python, so they are tested
with hand-built rotation matrices -- no camera or webcam involved. The OpenCV
``solvePnP`` estimator is exercised with a synthetic round-trip : we rotate the
canonical model, project it, feed the projected landmarks back and check that
the recovered angles match the applied rotation's *convention*.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.dynamics.headpose import (
    HeadPose,
    HeadPoseTracker,
    classify_direction,
    rotation_matrix_to_euler,
)

D = 180.0 / math.pi


def rx(deg: float) -> np.ndarray:
    a = deg / D
    return np.array(
        [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]
    )


def ry(deg: float) -> np.ndarray:
    a = deg / D
    return np.array(
        [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    )


def rz(deg: float) -> np.ndarray:
    a = deg / D
    return np.array(
        [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    )


class TestRotationMatrixToEuler:
    def test_identity_is_neutral(self):
        pitch, yaw, roll = rotation_matrix_to_euler(np.eye(3))
        assert pitch == pytest.approx(0.0, abs=1e-6)
        assert yaw == pytest.approx(0.0, abs=1e-6)
        assert roll == pytest.approx(0.0, abs=1e-6)

    def test_nod_down_is_positive_pitch(self):
        pitch, yaw, roll = rotation_matrix_to_euler(rx(20))
        assert pitch == pytest.approx(20.0, abs=1e-6)
        assert yaw == pytest.approx(0.0, abs=1e-6)
        assert roll == pytest.approx(0.0, abs=1e-6)

    def test_nod_up_is_negative_pitch(self):
        pitch, yaw, roll = rotation_matrix_to_euler(rx(-15))
        assert pitch == pytest.approx(-15.0, abs=1e-6)

    def test_turn_left_is_positive_yaw(self):
        pitch, yaw, roll = rotation_matrix_to_euler(ry(25))
        assert yaw == pytest.approx(25.0, abs=1e-6)
        assert pitch == pytest.approx(0.0, abs=1e-6)

    def test_turn_right_is_negative_yaw(self):
        pitch, yaw, roll = rotation_matrix_to_euler(ry(-18))
        assert yaw == pytest.approx(-18.0, abs=1e-6)

    def test_tilt_left_is_positive_roll(self):
        pitch, yaw, roll = rotation_matrix_to_euler(rz(12))
        assert roll == pytest.approx(12.0, abs=1e-6)
        assert yaw == pytest.approx(0.0, abs=1e-6)

    def test_combined_pose_recovers_all_axes(self):
        # Composite: turn left 12, nod down 10, tilt left 8. Small-angle
        # crosstalk between yaw and pitch is expected; stay within 2 degrees.
        pitch, yaw, roll = rotation_matrix_to_euler(ry(12) @ rx(10) @ rz(8))
        assert pitch == pytest.approx(10.0, abs=2.0)
        assert yaw == pytest.approx(12.0, abs=2.0)
        assert roll == pytest.approx(8.0, abs=2.0)


class TestClassifyDirection:
    def test_neutral_within_bounds(self):
        assert classify_direction(0.0, 0.0, 20, 20, 25, 25) == "NEUTRAL"
        assert classify_direction(10.0, 0.0, 20, 20, 25, 25) == "NEUTRAL"
        assert classify_direction(0.0, -20.0, 20, 20, 25, 25) == "NEUTRAL"

    def test_pitch_above_threshold_is_down(self):
        assert classify_direction(30.0, 0.0, 20, 20, 25, 25) == "DOWN"

    def test_pitch_below_negative_threshold_is_up(self):
        assert classify_direction(-25.0, 0.0, 20, 20, 25, 25) == "UP"

    def test_yaw_above_threshold_is_left(self):
        assert classify_direction(0.0, 30.0, 20, 20, 25, 25) == "LEFT"

    def test_yaw_below_negative_threshold_is_right(self):
        assert classify_direction(0.0, -30.0, 20, 20, 25, 25) == "RIGHT"

    def test_two_axes_combine(self):
        assert classify_direction(30.0, 30.0, 20, 20, 25, 25) == "DOWN-LEFT"
        assert classify_direction(-25.0, -28.0, 20, 20, 25, 25) == "UP-RIGHT"

    def test_threshold_is_inclusive(self):
        assert classify_direction(20.0, 0.0, 20, 20, 25, 25) == "DOWN"
        assert classify_direction(0.0, 25.0, 20, 20, 25, 25) == "LEFT"


def make_tracker(
    duration: float = 5.0, alpha: float = 0.4, gap: float = 1.0
) -> HeadPoseTracker:
    return HeadPoseTracker(
        pitch_down_threshold=20.0,
        pitch_up_threshold=20.0,
        yaw_left_threshold=25.0,
        yaw_right_threshold=25.0,
        duration_seconds=duration,
        smoothing_alpha=alpha,
        max_gap_seconds=gap,
    )


class TestHeadPoseTracker:
    def test_neutral_feed_never_accumulates(self):
        tracker = make_tracker()
        out = None
        for i in range(50):
            out = tracker.feed((0.0, 0.0, 0.0), i * 0.1)
        assert out.direction == "NEUTRAL"
        assert out.duration == 0.0
        assert not out.sustained

    def test_sustained_down_after_duration(self):
        tracker = make_tracker(duration=5.0)
        sustained = None
        for i in range(80):  # 8 s of a 30-deg pitch-down
            sustained = tracker.feed((30.0, 0.0, 0.0), i * 0.1)
        assert sustained.direction == "DOWN"
        assert sustained.sustained
        assert sustained.duration == pytest.approx(7.9, abs=0.1)

    def test_short_deviation_never_sustained(self):
        tracker = make_tracker(duration=5.0)
        out = None
        for i in range(10):  # only 1 s of deviation
            out = tracker.feed((30.0, 0.0, 0.0), i * 0.1)
        assert out.direction == "DOWN"
        assert not out.sustained

    def test_recovering_to_neutral_resets(self):
        tracker = make_tracker(duration=2.0)
        for i in range(30):
            tracker.feed((30.0, 0.0, 0.0), i * 0.1)
        reset = tracker.feed((0.0, 0.0, 0.0), 4.0)
        assert reset.direction == "NEUTRAL"
        assert reset.duration == 0.0
        assert not reset.sustained

    def test_gap_longer_than_max_gap_restarts_episode(self):
        tracker = make_tracker(gap=1.0)
        tracker.feed((30.0, 0.0, 0.0), 0.0)
        tracker.feed((30.0, 0.0, 0.0), 1.0)
        out = tracker.feed((30.0, 0.0, 0.0), 3.0)  # 2 s gap > max gap
        assert out.direction == "DOWN"
        assert out.duration == 0.0  # fresh episode, not counting the gap
        out = tracker.feed((30.0, 0.0, 0.0), 4.0)
        assert out.duration == pytest.approx(1.0, abs=1e-6)

    def test_none_input_resets_state(self):
        tracker = make_tracker()
        for i in range(20):
            tracker.feed((30.0, 0.0, 0.0), i * 0.1)
        out = tracker.feed(None, 3.0)
        assert out.direction is None
        assert out.duration == 0.0
        fresh = tracker.feed((10.0, 0.0, 0.0), 3.1)  # back to near-neutral
        assert fresh.duration == 0.0
        assert fresh.sustained is False

    def test_accepts_headpose_object(self):
        tracker = make_tracker()
        out = tracker.feed(HeadPose(pitch=30.0, yaw=0.0, roll=0.0), 0.0)
        assert out.direction == "DOWN"

    def test_ema_smooths_spikes(self):
        tracker = make_tracker(alpha=0.4, duration=10.0)
        first = tracker.feed((40.0, 0.0, 0.0), 0.0)
        assert first.pitch == pytest.approx(40.0, abs=1e-6)
        # A sudden return to neutral is pulled back slowly by the EMA.
        second = tracker.feed((0.0, 0.0, 0.0), 0.1)
        assert second.pitch == pytest.approx(24.0, abs=1e-6)  # 0.6*40
        assert second.direction == "DOWN"  # still above the 20-deg threshold
        third = tracker.feed((0.0, 0.0, 0.0), 0.2)
        assert third.direction == "NEUTRAL"  # 14.4 < 20


class TestHeadPoseEstimatorRoundTrip:
    """solvePnP round-trip: rotate the model, project, estimate, compare."""

    def _estimator(self):
        cv2 = pytest.importorskip("cv2")
        from src.vision.headpose import (
            FACE_MODEL_POINTS_3D,
            HEAD_POSE_LANDMARKS,
            HeadPoseEstimator,
        )

        return cv2, FACE_MODEL_POINTS_3D, HEAD_POSE_LANDMARKS, HeadPoseEstimator()

    def _landmarks(self, rotation: np.ndarray, w: int, h: int):
        cv2, model, indices, _ = self._estimator()
        K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        tvec = np.array([[0.0], [0.0], [100.0]])
        rotated = (model @ rotation.T).astype(np.float32)
        pixels, _ = cv2.projectPoints(
            rotated, np.zeros((3, 1), np.float64), tvec, K, dist
        )
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[list(indices)] = (pixels.reshape(-1, 2) / [w, h]).astype(np.float32)
        return landmarks

    def test_nod_down(self):
        cv2, _, __, est = self._estimator()
        pose = est.estimate(self._landmarks(rx(20), 640, 480), 640, 480)
        assert pose is not None
        assert pose.pitch == pytest.approx(20.0, abs=1.5)
        assert abs(pose.yaw) < 1.5
        assert abs(pose.roll) < 1.5

    def test_turn_left(self):
        _, _, _, est = self._estimator()
        pose = est.estimate(self._landmarks(ry(25), 640, 480), 640, 480)
        assert pose is not None
        assert pose.yaw == pytest.approx(25.0, abs=1.5)
        assert abs(pose.pitch) < 1.5

    def test_tilt_left(self):
        _, _, _, est = self._estimator()
        pose = est.estimate(self._landmarks(rz(15), 640, 480), 640, 480)
        assert pose is not None
        assert pose.roll == pytest.approx(15.0, abs=1.5)
        assert abs(pose.pitch) < 1.5

    def test_opposite_directions_flip_signs(self):
        _, _, _, est = self._estimator()
        down = est.estimate(self._landmarks(rx(-20), 640, 480), 640, 480)
        left = est.estimate(self._landmarks(ry(-20), 640, 480), 640, 480)
        assert down.pitch < 0  # nod up reads negative pitch
        assert left.yaw < 0  # turn right reads negative yaw

    def test_degenerate_all_zero_landmarks_returns_none(self):
        _, _, _, est = self._estimator()
        landmarks = np.zeros((478, 2), dtype=np.float32)
        assert est.estimate(landmarks, 640, 480) is None

    def test_too_few_landmarks_returns_none(self):
        _, _, _, est = self._estimator()
        assert est.estimate(np.zeros((100, 2), dtype=np.float32), 640, 480) is None

    def test_non_finite_landmarks_returns_none(self):
        _, _, _, est = self._estimator()
        landmarks = np.full((478, 2), np.nan, dtype=np.float32)
        assert est.estimate(landmarks, 640, 480) is None

    def test_zero_size_frame_returns_none(self):
        _, _, _, est = self._estimator()
        assert est.estimate(np.zeros((478, 2), dtype=np.float32), 0, 480) is None
