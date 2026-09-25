"""Geometric feature extraction: EAR, MAR and helpers.

This module is deliberately **OpenCV/MediaPipe-free**: it works on plain
``(N, 2)`` arrays of normalized landmark coordinates, so it is trivially
unit-testable and can be reused from any other face-landmark backend.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

_EPSILON = 1e-9

# Inner-lip landmark indices for the Mouth Aspect Ratio (MediaPipe FaceMesh).
MOUTH_VERTICAL_A = 13  # upper inner lip
MOUTH_VERTICAL_B = 14  # lower inner lip
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291


def euclidean(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Euclidean distance between two 2-D points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def eye_aspect_ratio(landmarks: NDArray[np.floating], eye_idx: Sequence[int]) -> float:
    """Compute the Eye Aspect Ratio for a single eye.

    Parameters
    ----------
    landmarks : (N, 2) array of normalized (x, y) coordinates.
    eye_idx   : the six landmark indices [p1..p6] as described in ``landmarks``.

    Returns
    -------
    float
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    The ratio is ~0.28-0.35 for a wide-open eye and approaches ~0 when the eye
    is closed, because the vertical distances collapse while the horizontal
    (corner-to-corner) distance barely changes.
    """
    pts = landmarks[np.asarray(eye_idx, dtype=int)]
    vertical_a = euclidean(pts[1], pts[5])
    vertical_b = euclidean(pts[2], pts[4])
    horizontal = euclidean(pts[0], pts[3])
    if horizontal < _EPSILON:
        return 0.0
    return float((vertical_a + vertical_b) / (2.0 * horizontal))


def average_eye_aspect_ratio(
    landmarks: NDArray[np.floating],
    left_eye_idx: Sequence[int],
    right_eye_idx: Sequence[int],
) -> float:
    """Average EAR over both eyes (normalizes left-right asymmetry)."""
    left = eye_aspect_ratio(landmarks, left_eye_idx)
    right = eye_aspect_ratio(landmarks, right_eye_idx)
    return float((left + right) / 2.0)


def mouth_aspect_ratio(landmarks: NDArray[np.floating]) -> float:
    """Mouth Aspect Ratio for yawn detection.

    MAR = |vertical| / |horizontal| where

        vertical   = distance between the inner upper and lower lip centers
        horizontal = distance between the two inner mouth corners

    A closed mouth has a small vertical gap (MAR ~ 0.05-0.2); a wide yawn
    drives MAR toward 1.0+ while the mouth-width stays roughly constant.
    """
    vertical_dist = euclidean(
        landmarks[MOUTH_VERTICAL_A], landmarks[MOUTH_VERTICAL_B]
    )
    horizontal_dist = euclidean(
        landmarks[MOUTH_LEFT_CORNER], landmarks[MOUTH_RIGHT_CORNER]
    )
    if horizontal_dist < _EPSILON:
        return 0.0
    return float(vertical_dist / horizontal_dist)


def to_pixel_coords(
    landmarks: NDArray[np.floating], width: int, height: int
) -> NDArray[np.intp]:
    """Map normalized (x, y) landmarks into integer pixel coordinates."""
    pixels = np.empty_like(landmarks, dtype=np.intp)
    pixels[:, 0] = np.clip(landmarks[:, 0] * width, 0, width - 1)
    pixels[:, 1] = np.clip(landmarks[:, 1] * height, 0, height - 1)
    return pixels


def landmark_bounding_box(
    landmarks: NDArray[np.floating], width: int, height: int
) -> tuple[int, int, int, int]:
    """Axis-aligned bounding box of all landmarks in pixel space (x, y, w, h)."""
    px = to_pixel_coords(landmarks, width, height)
    x0, y0 = int(px[:, 0].min()), int(px[:, 1].min())
    x1, y1 = int(px[:, 0].max()), int(px[:, 1].max())
    return x0, y0, x1 - x0, y1 - y0


__all__ = [
    "euclidean",
    "eye_aspect_ratio",
    "average_eye_aspect_ratio",
    "mouth_aspect_ratio",
    "MOUTH_VERTICAL_A",
    "MOUTH_VERTICAL_B",
    "MOUTH_LEFT_CORNER",
    "MOUTH_RIGHT_CORNER",
    "to_pixel_coords",
    "landmark_bounding_box",
]
