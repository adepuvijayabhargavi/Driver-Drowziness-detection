"""Unit tests for the camera auto-selection used by the ML data collector.

All tests use mocks/fakes -- no physical webcam is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.collect_ml_data import (
    FRAMES_TO_PROBE,
    CameraSelectionError,
    _is_effectively_black,
    _probe_camera,
    select_camera_source,
)


class FakeStream:
    """Minimal stand-in for src.camera.VideoStream (read/release/fps)."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.released = False
        self.fps = 30.0

    def read(self):
        if not self._frames:
            return None
        return self._frames.pop(0)

    def release(self):
        self.released = True


def _noop(*_args, **_kwargs):
    return None


class TestSelectCameraSource:
    def test_auto_scan_picks_first_working_index(self):
        calls = []

        def probe(idx):
            calls.append(idx)
            if idx == 1:
                return True, FakeStream([]), "working"
            return False, None, "no signal"

        source, stream = select_camera_source(
            None, probe=probe, max_index=3, status=_noop
        )
        assert source == 1
        assert stream is not None
        assert calls == [0, 1]

    def test_prints_required_startup_messages(self):
        lines = []

        def probe(idx):
            return (True, FakeStream([]), "working") if idx == 1 else (False, None, "no signal")

        select_camera_source(None, probe=probe, max_index=3, status=lines.append)
        assert lines == [
            "Testing camera 0...",
            "Camera 0: no signal (no signal)",
            "Testing camera 1...",
            "Camera 1: working",
            "Using camera source: 1",
        ]

    def test_raises_with_tested_indexes_when_none_work(self):
        with pytest.raises(CameraSelectionError) as exc_info:
            select_camera_source(
                None,
                probe=lambda i: (False, None, "black/empty frames"),
                max_index=2,
                status=_noop,
            )
        message = str(exc_info.value)
        assert "No working camera found" in message
        assert "0" in message and "1" in message

    def test_honours_explicit_preferred_index(self):
        used = []

        def probe(idx):
            used.append(idx)
            return True, FakeStream([]), "working"

        source, _stream = select_camera_source(
            3, probe=probe, max_index=5, status=_noop
        )
        assert source == 3
        assert used == [3]

    def test_reports_file_source_phrasing(self):
        lines = []

        def probe(src):
            return True, FakeStream([]), "working"

        select_camera_source("clip.mp4", probe=probe, status=lines.append)
        assert "Testing source clip.mp4..." in lines
        assert lines[-1] == "Using video source: clip.mp4"


class TestProbeCamera:
    def test_warmup_black_then_real_frame_counts_working(self, monkeypatch):
        import src.camera

        black = np.zeros((64, 64, 3), dtype=np.uint8)
        real = np.full((64, 64, 3), 120, dtype=np.uint8)
        fake = FakeStream([black, black, real])
        monkeypatch.setattr(src.camera, "VideoStream", lambda _config: fake)

        ok, stream, reason = _probe_camera(
            0, width=640, height=480, require_live=True
        )
        assert ok
        assert stream is fake
        assert reason == "working"

    def test_all_black_frames_rejected(self, monkeypatch):
        import src.camera

        black = np.zeros((64, 64, 3), dtype=np.uint8)
        fake = FakeStream([black] * FRAMES_TO_PROBE)
        monkeypatch.setattr(src.camera, "VideoStream", lambda _config: fake)

        ok, stream, reason = _probe_camera(
            0, width=640, height=480, require_live=True
        )
        assert not ok
        assert stream is None
        assert "black" in reason
        assert fake.released

    def test_read_failure_rejected(self, monkeypatch):
        import src.camera

        fake = FakeStream([])
        monkeypatch.setattr(src.camera, "VideoStream", lambda _config: fake)

        ok, _stream, reason = _probe_camera(
            0, width=640, height=480, require_live=True
        )
        assert not ok
        assert "read()" in reason

    def test_open_error_rejected(self, monkeypatch):
        import src.camera

        def raise_open(_config):
            raise RuntimeError("camera busy")

        monkeypatch.setattr(src.camera, "VideoStream", raise_open)
        ok, _stream, reason = _probe_camera(
            0, width=640, height=480, require_live=True
        )
        assert not ok
        assert "could not open" in reason

    def test_video_file_accepts_first_frame(self, monkeypatch):
        import src.camera

        frame = np.zeros((64, 64, 3), dtype=np.uint8)  # dark frame is OK for files
        fake = FakeStream([frame])
        monkeypatch.setattr(src.camera, "VideoStream", lambda _config: fake)

        ok, stream, _reason = _probe_camera(
            "clip.mp4", width=640, height=480, require_live=False
        )
        assert ok
        assert stream is fake


class TestBlackDetection:
    def test_black_frame_detected(self):
        assert _is_effectively_black(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_real_image_not_detected(self):
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        assert not _is_effectively_black(frame)
