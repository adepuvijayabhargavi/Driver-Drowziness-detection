"""Full-screen visual warnings drawn on top of the annotated frame."""

from __future__ import annotations

import numpy as np

from src.vision.renderer import COLOR_RED, COLOR_YELLOW, _cv2


def draw_fullscreen_warning(
    frame: np.ndarray,
    message: str,
    color: tuple[int, int, int] = COLOR_RED,
) -> np.ndarray:
    """Red-hued flash overlay with a large centered warning message."""
    cv2 = _cv2()
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), color, cv2.FILLED)
    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = message.upper()
    scale = 1.4 if w >= 800 else 1.0
    thickness = 3 if w >= 800 else 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cx = (w - tw) // 2
    cy = (h + th) // 2 - 20
    cv2.putText(frame, text, (cx, cy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    sub = "PULL OVER AND REST SAFELY" if "DROWSY" in message else "STAY FOCUSED"
    (sw, sh), _ = cv2.getTextSize(sub, font, 0.8, 2)
    cv2.putText(
        frame,
        sub,
        ((w - sw) // 2, cy + 30),
        font,
        0.8,
        COLOR_YELLOW,
        2,
        cv2.LINE_AA,
    )
    return frame


__all__ = ["draw_fullscreen_warning"]
