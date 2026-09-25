"""Tests for EAR / MAR geometry math."""

from __future__ import annotations

import numpy as np
import pytest

from src.dynamics.metrics import (
    average_eye_aspect_ratio,
    eye_aspect_ratio,
    mouth_aspect_ratio,
    to_pixel_coords,
)
from src.vision.landmarks import LEFT_EYE_IDX, RIGHT_EYE_IDX
from tests.conftest import face_with_ear, face_with_mar


class TestEyeAspectRatio:
    def test_open_eye_has_large_ear(self, open_face):
        ear = average_eye_aspect_ratio(open_face, LEFT_EYE_IDX, RIGHT_EYE_IDX)
        assert ear == pytest.approx(0.30, abs=1e-4)

    def test_closed_eye_has_small_ear(self, closed_face):
        ear = average_eye_aspect_ratio(closed_face, LEFT_EYE_IDX, RIGHT_EYE_IDX)
        assert ear < 0.10

    def test_ear_decreases_when_eye_closes(self, open_face, closed_face):
        open_ear = average_eye_aspect_ratio(open_face, LEFT_EYE_IDX, RIGHT_EYE_IDX)
        closed_ear = average_eye_aspect_ratio(closed_face, LEFT_EYE_IDX, RIGHT_EYE_IDX)
        assert open_ear > closed_ear

    def test_exact_formula(self):
        # EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p3-horizontal|)
        # With symmetric construction EAR == h/w exactly, verified numerically.
        for ear_target in (0.05, 0.1, 0.2, 0.3, 0.4):
            lm = face_with_ear(ear_target)
            left = eye_aspect_ratio(lm, LEFT_EYE_IDX)
            right = eye_aspect_ratio(lm, RIGHT_EYE_IDX)
            assert left == pytest.approx(ear_target, abs=1e-4)
            assert right == pytest.approx(ear_target, abs=1e-4)
            avg = average_eye_aspect_ratio(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX)
            assert avg == pytest.approx(ear_target, abs=1e-4)

    def test_zero_horizontal_distance_is_safe(self):
        lm = np.zeros((478, 2), dtype=np.float32)  # all points coincide
        assert eye_aspect_ratio(lm, LEFT_EYE_IDX) == 0.0

    def test_directional_symmetry(self):
        lm = face_with_ear(0.3)
        # flipping vertically should not change EAR
        flipped = lm.copy()
        flipped[:, 1] = 1.0 - flipped[:, 1]
        a = average_eye_aspect_ratio(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX)
        b = average_eye_aspect_ratio(flipped, LEFT_EYE_IDX, RIGHT_EYE_IDX)
        assert a == pytest.approx(b, abs=1e-4)


class TestMouthAspectRatio:
    def test_closed_mouth_small_mar(self, open_face):
        assert mouth_aspect_ratio(open_face) == pytest.approx(0.0833, abs=1e-3)

    def test_wide_open_mouth_high_mar(self, yawn_face):
        mar = mouth_aspect_ratio(yawn_face)
        assert mar == pytest.approx(0.8, abs=1e-3)
        assert mar > 0.5  # beyond the yawn threshold

    def test_mar_increases_with_gap(self):
        mars = [mouth_aspect_ratio(face_with_mar(m)) for m in (0.2, 0.5, 0.9)]
        assert mars == sorted(mars)

    def test_zero_width_is_safe(self):
        lm = np.zeros((478, 2), dtype=np.float32)
        assert mouth_aspect_ratio(lm) == 0.0


class TestGeometryHelpers:
    def test_to_pixel_coords_scales_and_clamps(self):

        lm = face_with_ear(0.3)
        px = to_pixel_coords(lm, 640, 480)
        assert px.shape == (478, 2)
        assert px.dtype.kind in "iu"
        assert px.min() >= 0
        assert px[:, 0].max() <= 639
        assert px[:, 1].max() <= 479
        # Landmark 33 is the right-eye outer corner at (0.42, 0.5) in this fixture.
        assert abs(px[33][0] - 0.42 * 640) <= 1
        assert abs(px[33][1] - 240) <= 1

    def test_euclidean(self):
        from src.dynamics.metrics import euclidean

        assert euclidean((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_ear_scale_invariance(self):
        # Scaling all coordinates must not change ratios.
        lm = face_with_ear(0.3) * 100.0  # outside [0,1] but ratios are invariant
        ear = average_eye_aspect_ratio(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX)
        assert ear == pytest.approx(0.30, abs=1e-3)
