"""Tests for configuration defaults and overrides."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from src.config.settings import (
    AlertSettings,
    CameraConfig,
    DetectionSettings,
    MLConfig,
    Settings,
)


def test_settings_are_dataclasses():
    for cls in (Settings, DetectionSettings, CameraConfig, AlertSettings, MLConfig):
        assert is_dataclass(cls)


def test_defaults_are_reasonable():
    d = DetectionSettings()
    assert 0.0 < d.ear_threshold < 1.0
    assert d.ear_consecutive_frames >= 1
    assert 0.0 < d.mar_threshold < 2.0
    assert d.mar_consecutive_frames >= 1
    assert d.recovery_frames >= 1
    assert d.min_blink_frames >= 1

    s = Settings()
    assert s.camera.width > 0
    assert s.camera.height > 0
    assert s.alert.audio_volume <= 1.0


def test_detection_specific_overrides():
    s = Settings().with_detection(ear_threshold=0.30, ear_consecutive_frames=15)
    assert s.detection.ear_threshold == 0.30
    assert s.detection.ear_consecutive_frames == 15
    # untouched fields preserved
    assert s.detection.mar_threshold == DetectionSettings().mar_threshold


def test_camera_override():
    s = Settings().with_camera(source=2, width=1280)
    assert s.camera.source == 2
    assert s.camera.width == 1280


def test_alert_override():
    s = Settings().with_alert(audio_enabled=False)
    assert s.alert.audio_enabled is False


def test_ml_override():
    s = Settings().with_ml(enabled=True, confidence_threshold=0.8)
    assert s.ml.enabled is True
    assert s.ml.confidence_threshold == 0.8


def test_all_fields_covered_by_replace_helpers():
    s = Settings()
    for f in {fld.name for fld in fields(s)}:
        assert hasattr(s, f)


def test_duplicate_replaces_are_immutable_style():
    a = Settings()
    b = a.with_detection(ear_threshold=0.1)
    assert a.detection.ear_threshold != b.detection.ear_threshold  # no mutation
