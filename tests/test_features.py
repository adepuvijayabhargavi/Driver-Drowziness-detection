"""Tests for the ML feature extractor."""

from __future__ import annotations

import numpy as np

from src.ml.features import FeatureExtractor, extract_batch_features


class TestFeatureExtractor:
    def test_dimension_and_names(self):
        assert FeatureExtractor.dimension() == 10
        assert len(FeatureExtractor.feature_names()) == 10

    def test_features_shape(self):
        fe = FeatureExtractor(window=10)
        for _ in range(15):
            vec = fe.update_and_features(ear=0.3, mar=0.1)
            assert vec.shape == (10,)
            assert vec.dtype == np.float32

    def test_nan_until_window_warm(self):
        fe = FeatureExtractor(window=4)
        first = fe.update_and_features(0.3, 0.1)
        assert np.isnan(first).all()  # single sample -> undefined stats

    def test_closed_fraction_high_when_eyes_closed(self):
        fe = FeatureExtractor(ear_threshold=0.25)
        for _ in range(20):
            fe.update(ear=0.05, mar=0.1)
        vec = fe.features()
        assert vec[7] > 0.9      # eye_closed_fraction
        assert vec[0] < 0.1      # ear_mean

    def test_open_fraction_reflects_mouth(self):
        fe = FeatureExtractor(mar_threshold=0.55)
        for _ in range(20):
            fe.update(ear=0.3, mar=0.8)   # sustained open mouth
        vec = fe.features()
        assert vec[8] > 0.9      # mouth_open_fraction
        assert vec[5] > 0.7      # mar_max

    def test_window_limits_history(self):
        fe = FeatureExtractor(window=5)
        for _ in range(50):
            fe.update(ear=0.3, mar=0.1)
        assert len(fe._ear) == 5
        assert len(fe._mar) == 5

    def test_reset_clears(self):
        fe = FeatureExtractor(window=5)
        fe.update(0.3, 0.1)
        fe.reset()
        assert len(fe._ear) == 0
        assert len(fe._mar) == 0
        assert np.isnan(fe.features()).all()


class TestBatchExtraction:
    def test_shape_matches_sequence(self):
        t = 100
        ear = np.full(t, 0.3)
        mar = np.full(t, 0.1)
        out = extract_batch_features(ear, mar, window=15)
        assert out.shape == (t, 10)
        assert np.isnan(out[:14]).all()  # window not warm yet
        assert not np.isnan(out[15:]).any()

    def test_drowsy_window_captured(self):
        # 60 frames: alert initial, then 30 closed-eye frames in the middle.
        ear = np.full(90, 0.30)
        ear[40:70] = 0.05
        mar = np.full(90, 0.10)
        out = extract_batch_features(ear, mar, window=30)
        mid = out[60]
        end = out[-1]
        assert mid[7] > 0.5      # window inside the closure
        assert mid[0] < 0.2      # depressed EAR mean
        assert end[7] < 0.4      # window shifted past recovery
