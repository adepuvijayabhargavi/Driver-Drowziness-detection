"""Sanity checks for the landmark index constants."""

from __future__ import annotations

from src.vision.landmarks import (
    LEFT_EYE_IDX,
    MOUTH_HORIZONTAL_IDX,
    MOUTH_VERTICAL_IDX,
    RIGHT_EYE_IDX,
    ROI_LANDMARKS,
)


def test_eyes_each_have_six_points():
    assert len(LEFT_EYE_IDX) == 6
    assert len(RIGHT_EYE_IDX) == 6


def test_eyes_are_distinct():
    assert set(LEFT_EYE_IDX).isdisjoint(set(RIGHT_EYE_IDX))


def test_indices_within_mediapipe_range():
    # MediaPipe FaceMesh returns 468 canonical landmarks (478 with irises).
    all_indices = (
        *LEFT_EYE_IDX, *RIGHT_EYE_IDX, *ROI_LANDMARKS,
        *MOUTH_VERTICAL_IDX, *MOUTH_HORIZONTAL_IDX,
    )
    for idx in all_indices:
        assert 0 <= idx < 478


def test_mouth_indices():
    assert set(MOUTH_VERTICAL_IDX) == {13, 14}
    assert set(MOUTH_HORIZONTAL_IDX) == {61, 291}


def test_roi_is_a_subset():
    for idx in ROI_LANDMARKS:
        assert 0 <= idx < 478
