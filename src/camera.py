"""Video capture helper.

A thin, fault-tolerant wrapper around OpenCV's VideoCapture that:
  * accepts a camera index, a video file path, its name starts with 0-9
  * requests a resolution/FPS and *reports* what the backend actually
    delivered (cameras frequently ignore requests).
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from src.config.settings import CameraConfig

logger = logging.getLogger(__name__)


class VideoStream:
    """Frame iterator over a webcam or video file."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        import cv2  # lazy import keeps the module importable without OpenCV

        self.config = config or CameraConfig()
        source = self.config.source
        is_camera = isinstance(source, int) or str(source).isdigit()

        if is_camera:
            self._cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(str(source))

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open video source {source!r}. Is the webcam in use "
                "by another app, or is the file path wrong?"
            )

        if is_camera:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            if self.config.target_fps > 0:
                self._cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.config.width
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.config.height
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 0 < fps < 240 else self.config.target_fps

        logger.info(
            "Video source %r open: %dx%d @ %.1f fps",
            source,
            self.width,
            self.height,
            self.fps,
        )

    def read(self) -> np.ndarray | None:
        """Read one BGR frame; returns None at end of file / on error."""
        ok, frame = self._cap.read()
        return frame if ok else None

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def release(self) -> None:
        if getattr(self, "_cap", None) is not None:
            self._cap.release()

    def __enter__(self) -> "VideoStream":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def open_stream(config: CameraConfig | None = None) -> VideoStream:
    return VideoStream(config)


__all__ = ["VideoStream", "open_stream"]
