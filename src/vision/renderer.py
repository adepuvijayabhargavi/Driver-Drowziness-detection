"""OpenCV rendering helpers for the annotated video output.

Kept separate from the detection logic so that everything that touches the
frame is easy to swap (e.g. for a Qt/flask frontend later).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.dynamics.metrics import to_pixel_coords
from src.vision.landmarks import (
    LEFT_EYE_ALL_IDX,
    MOUTH_OUTLINE_IDX,
    RIGHT_EYE_ALL_IDX,
)

# BGR colors
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 200, 0)
COLOR_YELLOW = (0, 200, 255)
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 128, 255)


def _cv2():
    import cv2  # lazy import: renderer can be imported without OpenCV

    return cv2


def draw_face_regions(
    frame: np.ndarray, landmarks: np.ndarray
) -> np.ndarray:
    """Overlay the eyes, mouth outline and their bounding polygon boxes."""
    cv2 = _cv2()
    h, w = frame.shape[:2]
    pts = to_pixel_coords(landmarks, w, h)

    def outline(indices: Sequence[int], color: tuple[int, int, int]) -> None:
        poly = pts[np.asarray(indices)].reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

    outline(LEFT_EYE_ALL_IDX, COLOR_GREEN)
    outline(RIGHT_EYE_ALL_IDX, COLOR_GREEN)
    outline(MOUTH_OUTLINE_IDX, COLOR_YELLOW)
    return frame


def draw_status_overlay(
    frame: np.ndarray,
    text_lines: Sequence[str],
    position: tuple[int, int] = (12, 24),
    color: tuple[int, int, int] = COLOR_WHITE,
    background: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Draw a small translucent panel with one text line per element."""
    cv2 = _cv2()
    pad = 6
    line_h = 22
    if background is not None:
        panel_height = 12 + line_h * len(text_lines)
        panel_width = 320
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (position[0] - pad, position[1] - 8),
            (position[0] + panel_width, position[1] - 8 + panel_height),
            background,
            cv2.FILLED,
        )
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    for i, line in enumerate(text_lines):
        y = position[1] + i * line_h
        cv2.putText(
            frame,
            line,
            (position[0], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def state_color(state: str) -> tuple[int, int, int]:
    """Map a state string to its alert color."""
    s = state.upper()
    if s == "DROWSY":
        return COLOR_RED
    if s == "YAWNING":
        return COLOR_ORANGE
    if s == "UNKNOWN":
        return COLOR_YELLOW
    return COLOR_GREEN


__all__ = [
    "draw_face_regions",
    "draw_status_overlay",
    "state_color",
    "COLOR_WHITE",
    "COLOR_GREEN",
    "COLOR_YELLOW",
    "COLOR_RED",
    "COLOR_ORANGE",
]
