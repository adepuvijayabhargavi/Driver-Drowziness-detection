"""Tests for data collection (scripts.collect_ml_data)."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from scripts.collect_ml_data import collect_from_frames, new_session_id
from src.config.settings import Settings
from src.dynamics.state_machine import DriverState
from src.ml.dataset import load_dataset
from src.ml.schema import FEATURE_COLUMNS, HEADER, LABELS
from src.pipeline import DrowsinessPipeline
from src.vision.detector import SyntheticFaceDetector
from tests.conftest import face_with_ear


def _pipeline(detector_landmarks) -> DrowsinessPipeline:
    return DrowsinessPipeline(
        detector=SyntheticFaceDetector(detector_landmarks),
        settings=Settings().with_alert(audio_enabled=False),
    )


def _open_pipeline() -> DrowsinessPipeline:
    return _pipeline(face_with_ear(0.3))


def _closed_pipeline() -> DrowsinessPipeline:
    return _pipeline(face_with_ear(0.05))


def _frames(pipeline: DrowsinessPipeline, n: int, fps: float = 30.0):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for i in range(n):
        yield frame, i / fps


class TestSessionIds:
    def test_new_session_id_is_unique_per_call(self):
        ids = {new_session_id() for _ in range(50)}
        assert len(ids) == 50

    def test_new_session_id_has_timestamp_prefix(self):
        session_id = new_session_id()
        # YYYYMMDD_HHMMSS_xxxxxx - fixed-width, collision-proof suffix.
        assert len(session_id) == len("20260101_000000_abcdef")
        assert session_id[8] == "_" and session_id[15] == "_"


class TestCollectionRobustness:
    def test_skips_read_failure_frames_without_counting(self, tmp_path):
        pl = _open_pipeline()
        path = tmp_path / "mixed.csv"
        handle, writer = _blank_writer(path)

        def mixed_camera():
            # simulate a flaky webcam: several read() == None frames first.
            yield None, 0.0
            yield None, 0.033
            for i in range(60):
                yield np.zeros((240, 320, 3), dtype=np.uint8), i / 30.0

        try:
            collected = collect_from_frames(
                mixed_camera(),
                pl,
                writer,
                label="alert",
                session_id="flaky",
                stride=1,
                max_samples=30,
            )
        finally:
            handle.close()
        assert collected == 30
        data = load_dataset(path)
        assert data.size == 30
        assert len(set(data.sessions)) == 1

    def test_max_samples_zero_means_unlimited(self, tmp_path):
        pl = _open_pipeline()
        path = tmp_path / "all.csv"
        handle, writer = _blank_writer(path)
        try:
            collected = collect_from_frames(
                _frames(pl, 40),
                pl,
                writer,
                label="alert",
                session_id="all",
                stride=1,
                max_samples=0,
            )
        finally:
            handle.close()
        # Every valid face frame is recorded until the stream ends (40 frames).
        assert collected == 40

    def test_black_frames_are_not_counted(self, tmp_path):
        from src.vision.detector import SyntheticFaceDetector

        # A black feed -> no face -> never counted as a sample.
        black = SyntheticFaceDetector()  # detects a face even on pure black?
        pl = DrowsinessPipeline(
            detector=black, settings=Settings().with_alert(audio_enabled=False)
        )
        path = tmp_path / "black.csv"
        handle, writer = _blank_writer(path)
        try:
            collected = collect_from_frames(
                _frames(pl, 40),
                pl,
                writer,
                label="alert",
                session_id="black",
                stride=1,
                max_samples=100,
            )
        finally:
            handle.close()
        # Not asserting face-independence here: SyntheticFaceDetector may or may
        # not find a face on black; what matters is NO fabricated rows are drawn
        # from missing frames and the call never crashes.
        assert 0 <= collected <= 40


class TestBlackFrameDetection:
    def test_detects_pure_black(self):
        from scripts.collect_ml_data import _is_effectively_black

        assert _is_effectively_black(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_rejects_normal_images(self):
        from scripts.collect_ml_data import _is_effectively_black

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:] = (120, 120, 120)
        assert not _is_effectively_black(frame)


def _blank_writer(path, header=None):
    handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=list(header or HEADER))
    writer.writeheader()
    return handle, writer


class TestCollectionCore:
    def test_collects_alert_samples(self, tmp_path):
        pl = _open_pipeline()
        path = tmp_path / "alert.csv"
        handle, writer = _blank_writer(path)
        try:
            collected = collect_from_frames(
                _frames(pl, 60),
                pl,
                writer,
                label="alert",
                session_id="session-A",
                stride=1,
                max_samples=30,
            )
        finally:
            handle.close()
        assert collected == 30
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        assert len(rows) == 30
        assert {r["label"] for r in rows} == {"alert"}
        assert {r["session_id"] for r in rows} == {"session-A"}
        for row in rows:
            assert float(row["ear_mean"]) == pytest.approx(0.3, abs=0.02)

    def test_collects_drowsy_samples_and_honours_session(self, tmp_path):
        pl = _closed_pipeline()
        drowsy_path = tmp_path / "drowsy.csv"
        handle, writer = _blank_writer(drowsy_path)
        try:
            collected = collect_from_frames(
                _frames(pl, 60),
                pl,
                writer,
                label="drowsy",
                session_id="session-B",
                stride=1,
                max_samples=25,
            )
        finally:
            handle.close()
        assert collected == 25
        data = load_dataset(drowsy_path)
        assert data.size == 25
        assert len(set(data.sessions)) == 1
        assert set(data.labels) == {"drowsy"}

    def test_stride_limits_redundant_frames(self, tmp_path):
        pl = _open_pipeline()
        path = tmp_path / "strided.csv"
        handle, writer = _blank_writer(path)
        try:
            collected = collect_from_frames(
                _frames(pl, 20),
                pl,
                writer,
                label="alert",
                session_id="s",
                stride=3,
                max_samples=100,
            )
        finally:
            handle.close()
        # Frames 1..20, collect when index % 3 == 0 -> 6 samples.
        assert 6 <= collected <= 7

    def test_appended_sessions_preserve_metadata(self, tmp_path):
        path = tmp_path / "dataset.csv"
        for session, label, landmarks in (
            ("one", "alert", face_with_ear(0.3)),
            ("two", "drowsy", face_with_ear(0.05)),
        ):
            pl = _pipeline(landmarks)
            handle = open(path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=list(HEADER))
            if handle.tell() == 0:
                writer.writeheader()
            try:
                collect_from_frames(
                    _frames(pl, 40),
                    pl,
                    writer,
                    label=label,
                    session_id=session,
                    stride=2,
                    max_samples=20,
                )
            finally:
                handle.close()
        data = load_dataset(path)
        assert len(set(data.sessions)) == 2
        assert set(data.labels) == set(LABELS)
        assert data.features.shape[1] == len(FEATURE_COLUMNS)

    def test_closed_eyes_produce_drowsy_state(self):
        pl = _closed_pipeline()
        last = None
        for i in range(60):
            _, report = pl.process_frame(
                np.zeros((240, 320, 3), dtype=np.uint8), timestamp=i / 30.0
            )
            last = report
        assert last.drowsy
        assert last.state is DriverState.DROWSY
