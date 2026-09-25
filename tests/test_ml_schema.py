"""Tests for the shared ML feature schema (src.ml.schema)."""

from __future__ import annotations

import numpy as np
import pytest

from src.dynamics.state_machine import DriverState, FrameReport
from src.ml.schema import (
    FEATURE_COLUMNS,
    HEADER,
    LABELS,
    N_FEATURES,
    feature_names,
    feature_row,
    feature_vector_from_report,
    validate_feature_order,
)


def _full_report() -> FrameReport:
    """A report where every schema feature is present and finite."""
    return FrameReport(
        frame_index=1,
        timestamp=1.0,
        face_present=True,
        ear=0.30,
        mar=0.10,
        state=DriverState.ALERT,
        drowsy=False,
        yawning=False,
        eye_closed_frames=0,
        mouth_open_frames=0,
        blink_count=0,
        yawn_count=0,
        blink_rate_per_min=0.0,
        no_face_frames=0,
        ear_left=0.29,
        ear_right=0.31,
        eye_closed_duration=0.0,
        perclos=12.0,
        yawn_duration=0.0,
        head_pitch=2.0,
        head_yaw=-3.0,
        head_roll=1.0,
    )


class TestSchemaDefinition:
    def test_feature_columns_exact_expected(self):
        assert FEATURE_COLUMNS == (
            "ear_left",
            "ear_right",
            "ear_mean",
            "mar",
            "perclos",
            "eye_closed_duration",
            "yawn_duration",
            "pitch",
            "yaw",
            "roll",
        )

    def test_count_and_names_consistency(self):
        assert N_FEATURES == 10
        assert len(feature_names()) == N_FEATURES
        assert feature_names() == list(FEATURE_COLUMNS)

    def test_header_layout(self):
        assert HEADER == (
            "timestamp",
            "session_id",
        ) + FEATURE_COLUMNS + ("label",)

    def test_labels(self):
        assert LABELS == ("alert", "drowsy")

    def test_validate_feature_order_accepts_canonical(self):
        assert validate_feature_order(feature_names()) is True

    def test_validate_feature_order_rejects_reordered(self):
        assert validate_feature_order(list(FEATURE_COLUMNS)[::-1]) is False

    def test_validate_feature_order_rejects_subset(self):
        assert validate_feature_order(list(FEATURE_COLUMNS)[:3]) is False


class TestFeatureVector:
    def test_full_report_yields_finite_vector(self):
        vector = feature_vector_from_report(_full_report())
        assert vector.shape == (N_FEATURES,)
        assert vector.dtype == np.float32
        assert not np.isnan(vector).any()

    def test_order_matches_schema(self):
        report = _full_report()
        vector = feature_vector_from_report(report)
        expect = np.asarray(
            [
                0.29,              # ear_left
                0.31,              # ear_right
                0.30,              # ear_mean (report.ear)
                0.10,              # mar
                12.0,              # perclos
                0.0,               # eye_closed_duration
                0.0,               # yawn_duration
                2.0,               # pitch -> report.head_pitch
                -3.0,              # yaw -> report.head_yaw
                1.0,               # roll -> report.head_roll
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(vector, expect, atol=1e-6)

    def test_unavailable_fields_become_nan(self):
        report = _full_report()
        report.ear_left = None
        report.head_pitch = None
        vector = feature_vector_from_report(report)
        assert np.isnan(vector[0])       # ear_left
        assert np.isnan(vector[7])       # pitch
        assert not np.isnan(vector[2])   # ear_mean still present

    def test_default_report_has_nans_not_fabricated(self):
        report = _full_report()
        report.ear_left = None
        report.ear_right = None
        report.perclos = None
        vector = feature_vector_from_report(report)
        assert np.isnan(vector[0])
        assert np.isnan(vector[1])
        assert np.isnan(vector[4])
        assert vector[2] == 0.30


class TestFeatureRow:
    def test_row_matches_header(self):
        report = _full_report()
        row = feature_row(
            report, timestamp=1.5, session_id="sess-1", label="alert"
        )
        assert list(row) == list(HEADER)
        assert row["timestamp"] == 1.5
        assert row["session_id"] == "sess-1"
        assert row["label"] == "alert"
        assert row["ear_left"] == "0.29"

    def test_nan_features_written_as_empty_string(self):
        report = _full_report()
        report.head_yaw = None
        row = feature_row(
            report, timestamp=0.0, session_id="s1", label="drowsy"
        )
        assert row["yaw"] == ""

    def test_invalid_label_rejected(self):
        with pytest.raises(ValueError):
            feature_row(
                _full_report(),
                timestamp=0.0,
                session_id="s1",
                label="sleeping",
            )
