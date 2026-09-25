"""Shared fixtures: synthetic landmark skeletons and helpers.

The helpers build a ``478 x 2`` normalized landmark array where only the
eye/mouth landmarks matter, so the exact EAR/MAR are analytically known.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pytest

from src.vision.landmarks import LEFT_EYE_IDX, RIGHT_EYE_IDX

# Sensible defaults mirroring src/config/settings.py.
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.55


def eye_width() -> float:
    return 0.08


def _set_eye(lm: np.ndarray, indices, half_width: float, half_height: float) -> None:
    """Symmetric eye: EAR = half_height / half_width exactly."""
    layout = (
        (-half_width, 0.0),
        (-half_width / 2, half_height),
        (half_width / 2, half_height),
        (half_width, 0.0),
        (half_width / 2, -half_height),
        (-half_width / 2, -half_height),
    )
    for i, (dx, dy) in enumerate(layout):
        lm[indices[i]] = (0.5 + dx, 0.5 + dy)


def face_with_ear(ear: float) -> np.ndarray:
    """Skeleton with both eyes at the requested EAR (clamped to a sane range)."""
    lm = np.zeros((478, 2), dtype=np.float32)
    h = ear * eye_width()
    _set_eye(lm, RIGHT_EYE_IDX, eye_width(), h)
    _set_eye(lm, LEFT_EYE_IDX, eye_width(), h)
    # Closed mouth.
    lm[13] = (0.5, 0.5 + 0.015)
    lm[14] = (0.5, 0.5 - 0.015)
    lm[61] = (0.5 - 0.18, 0.5)
    lm[291] = (0.5 + 0.18, 0.5)
    return lm


def face_with_mar(mar: float) -> np.ndarray:
    """Skeleton with open eyes and the requested MAR (mouth gap / 0.36)."""
    lm = face_with_ear(0.30)
    gap = 0.36 * mar
    lm[13] = (0.5, 0.5 + gap / 2)
    lm[14] = (0.5, 0.5 - gap / 2)
    return lm


@pytest.fixture
def open_face() -> np.ndarray:
    return face_with_ear(0.3)


@pytest.fixture
def closed_face() -> np.ndarray:
    return face_with_ear(0.05)


@pytest.fixture
def yawn_face() -> np.ndarray:
    return face_with_mar(0.8)


@pytest.fixture
def neutral_smiling_face() -> np.ndarray:
    return face_with_ear(0.3)


@pytest.fixture
def settings():
    from src.config.settings import DetectionSettings

    return DetectionSettings(
        ear_threshold=EAR_THRESHOLD,
        mar_threshold=MAR_THRESHOLD,
        ear_consecutive_frames=30,
        mar_consecutive_frames=30,
        recovery_frames=10,
    )


__all__ = [
    "face_with_ear",
    "face_with_mar",
    "EAR_THRESHOLD",
    "MAR_THRESHOLD",
]
