"""Tests for ML integration inside the real-time pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.config.settings import Settings
from src.pipeline import DrowsinessPipeline
from src.vision.detector import SyntheticFaceDetector
from tests.conftest import face_with_ear


def _blank(shape=(240, 320, 3)) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


class _FakeClassifier:
    """Minimal classifier double: just confirms the pipeline fusion path."""

    enabled = True

    def evaluate(self, vector):
        return ("DROWSY", 0.94)


class TestPerEyeEar:
    def test_per_eye_ear_set_by_pipeline(self):
        pl = DrowsinessPipeline(
            detector=SyntheticFaceDetector(face_with_ear(0.3)),
            settings=Settings().with_alert(audio_enabled=False),
        )
        _, report = pl.process_frame(_blank(), timestamp=0.0)
        assert report.ear_left == pytest.approx(0.3, abs=1e-5)
        assert report.ear_right == pytest.approx(0.3, abs=1e-5)
        assert report.ear == pytest.approx(0.3, abs=1e-5)

    def test_eye_closed_duration_accumulates(self):
        settings = Settings().with_alert(audio_enabled=False)
        pl = DrowsinessPipeline(
            detector=SyntheticFaceDetector(face_with_ear(0.05)),
            settings=settings,
        )
        last = None
        for i in range(30):
            _, last = pl.process_frame(_blank(), timestamp=i / 30.0)
        assert last.eye_closed_duration > 0.5


class TestMLOff:
    def test_no_classifier_report_fields_are_none(self):
        pl = DrowsinessPipeline(
            detector=SyntheticFaceDetector(face_with_ear(0.3)),
            settings=Settings().with_alert(audio_enabled=False),
        )
        _, report = pl.process_frame(_blank(), timestamp=0.0)
        assert report.ml_prediction is None
        assert report.ml_confidence is None
        assert pl.ml_stats == (0, 0)

    def test_missing_model_flag_does_not_crash_pipeline(self, tmp_path):
        settings = (
            Settings()
            .with_alert(audio_enabled=False)
            .with_ml(
                enabled=True,
                model_path=tmp_path / "no_such_model.joblib",
            )
        )
        pl = DrowsinessPipeline(
            detector=SyntheticFaceDetector(face_with_ear(0.3)),
            settings=settings,
        )
        assert pl.ml is not None
        assert not pl.ml.enabled
        _, report = pl.process_frame(_blank(), timestamp=0.0)
        assert report.ml_prediction is None
        assert pl.ml_stats[0] >= 1
        assert pl.ml_stats[1] == 0


class TestMLOn:
    def test_fusion_sets_report_fields_and_counts_hits(self):
        pl = DrowsinessPipeline(
            detector=SyntheticFaceDetector(face_with_ear(0.3)),
            settings=Settings().with_alert(audio_enabled=False),
            classifier=_FakeClassifier(),
        )
        _, report = pl.process_frame(_blank(), timestamp=0.0)
        assert report.ml_prediction == "DROWSY"
        assert report.ml_confidence == 0.94
        calls, hits = pl.ml_stats
        assert calls == 1
        assert hits == 1

    def test_low_confidence_does_not_count_as_hit(self):
        low = _FakeClassifier()
        low.evaluate = lambda vector: ("DROWSY", 0.40)  # type: ignore[method-assign]

        pl = DrowsinessPipeline(
            detector=SyntheticFaceDetector(face_with_ear(0.3)),
            settings=Settings().with_alert(audio_enabled=False),
            classifier=low,
        )
        pl.process_frame(_blank(), timestamp=0.0)
        assert pl.ml_stats == (1, 0)
