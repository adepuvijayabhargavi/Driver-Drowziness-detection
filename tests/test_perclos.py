"""Tests for the rolling-window PERCLOS calculator and its state-machine hookup."""

from __future__ import annotations

import pytest

from src.config.settings import DetectionSettings
from src.dynamics.perclos import PerclosCalculator
from src.dynamics.state_machine import DriverState, DrowsinessStateMachine


class TestPerclosCalculator:
    def test_zero_percent_when_never_closed(self):
        calc = PerclosCalculator(window_seconds=60.0, ear_threshold=0.25)
        for i in range(61):
            calc.update(0.30, float(i))
        assert calc.perclos == pytest.approx(0.0, abs=1e-6)
        assert calc.closed_time == pytest.approx(0.0, abs=1e-6)

    def test_hundred_percent_when_always_closed(self):
        calc = PerclosCalculator(window_seconds=60.0, ear_threshold=0.25)
        for i in range(61):
            calc.update(0.05, float(i))
        assert calc.perclos == pytest.approx(100.0, abs=1e-6)
        assert calc.closed_time == pytest.approx(60.0, abs=1e-6)

    def test_fifty_percent_alternating_closures(self):
        calc = PerclosCalculator(window_seconds=60.0, ear_threshold=0.25)
        for i in range(61):  # closed on odd seconds -> exactly half the time
            calc.update(0.05 if i % 2 == 1 else 0.30, float(i))
        assert calc.perclos == pytest.approx(50.0, abs=1e-6)

    def test_rolling_window_expiration(self):
        calc = PerclosCalculator(window_seconds=60.0, ear_threshold=0.25, max_gap_seconds=2.0)
        for i in range(61):
            calc.update(0.05, float(i))  # closed for the first 60 s
        assert calc.perclos == pytest.approx(100.0, abs=1e-6)
        # Re-open; once the last closed interval (ended t=60) ages out of the
        # window (now >= 120), PERCLOS must fall back to ~0%.
        for i in range(61, 121):
            calc.update(0.30, float(i))
        assert calc.perclos == pytest.approx(0.0, abs=1e-6)

    def test_invalid_ear_values_never_counted_as_closed(self):
        calc = PerclosCalculator(window_seconds=60.0, ear_threshold=0.25, max_gap_seconds=2.0)
        calc.update(0.05, 0.0)
        calc.update(0.05, 1.0)   # closed 0-1
        calc.update(0.05, 2.0)   # closed 1-2  -> closed_time = 2 s
        calc.update(float("nan"), 3.0)  # invalid -> ignored
        calc.update(None, 4.0)          # no face  -> ignored
        calc.update(float("inf"), 5.0)  # invalid -> ignored
        calc.update(-1.0, 6.0)          # invalid (negative) -> ignored
        calc.update(0.30, 7.0)   # gap 2-7s > max_gap -> interval dropped
        calc.update(0.30, 8.0)   # open 7-8
        assert calc.closed_time == pytest.approx(2.0, abs=1e-6)
        assert calc.monitoring_time == pytest.approx(3.0, abs=1e-6)
        assert calc.perclos == pytest.approx(2.0 / 3.0 * 100.0, abs=1e-6)

    def test_no_face_frames_are_not_closed_eyes(self):
        calc = PerclosCalculator(window_seconds=60.0, ear_threshold=0.25, max_gap_seconds=1.0)
        calc.update(0.05, 0.0)   # one closed sample, then face lost at 1-3 s
        calc.update(None, 1.0)
        calc.update(None, 2.0)
        calc.update(None, 3.0)
        calc.update(0.30, 4.0)   # 4 s gap > max_gap -> unobserved time dropped
        assert calc.closed_time == pytest.approx(0.0, abs=1e-6)
        assert calc.monitoring_time == pytest.approx(0.0, abs=1e-6)
        calc.update(0.30, 5.0)   # open 4-5
        assert calc.monitoring_time == pytest.approx(1.0, abs=1e-6)
        assert calc.perclos == pytest.approx(0.0, abs=1e-6)

    def test_variable_frame_intervals_supported(self):
        calc = PerclosCalculator(window_seconds=60.0, ear_threshold=0.25, max_gap_seconds=5.0)
        calc.update(0.05, 0.0)
        calc.update(0.05, 0.5)   # +0.5 closed
        calc.update(0.05, 2.0)   # +1.5 closed
        calc.update(0.30, 3.0)   # +1.0 open
        calc.update(0.30, 3.2)   # +0.2 open
        calc.update(0.05, 6.0)   # +2.8 closed
        assert calc.closed_time == pytest.approx(4.8, abs=1e-6)
        assert calc.monitoring_time == pytest.approx(6.0, abs=1e-6)
        assert calc.perclos == pytest.approx(4.8 / 6.0 * 100.0, abs=1e-6)

    def test_empty_window(self):
        calc = PerclosCalculator(window_seconds=60.0, ear_threshold=0.25)
        assert calc.perclos == 0.0
        assert calc.closed_time == 0.0
        assert calc.monitoring_time == 0.0
        assert calc.sample_count == 0
        # A single sample has nothing to compare yet -> still 0%.
        calc.update(0.30, 1.0)
        assert calc.perclos == 0.0
        assert calc.sample_count == 1

    def test_ear_threshold_is_strict(self):
        calc = PerclosCalculator(window_seconds=60.0, ear_threshold=0.25)
        calc.update(0.25, 0.0)   # exactly on the threshold -> NOT closed
        calc.update(0.25, 1.0)
        calc.update(0.30, 2.0)
        calc.update(0.05, 3.0)   # below the threshold -> closed
        calc.update(0.30, 4.0)
        assert calc.closed_time == pytest.approx(1.0, abs=1e-6)
        assert calc.perclos == pytest.approx(25.0, abs=1e-6)


def make_settings(**over) -> DetectionSettings:
    params = dict(
        ear_threshold=0.25,
        mar_threshold=0.55,
        ear_consecutive_frames=30,
        mar_consecutive_frames=30,
        min_blink_frames=2,
        recovery_frames=10,
    )
    params.update(over)
    return DetectionSettings(**params)


class TestStateMachinePerclosIntegration:
    def test_sustained_perclos_enters_drowsy_without_ear_streak(self):
        sm = DrowsinessStateMachine(
            make_settings(perclos_drowsy_threshold=60.0, perclos_drowsy_seconds=5.0), fps=30
        )
        last = None
        for i in range(80):  # 8 s of open eyes but PERCLOS pinned at 80%
            last = sm.update(0.30, 0.1, timestamp=i * 0.1, perclos=80.0)
        assert sm.eye_closed_frames == 0          # EAR path never tripped
        assert last.state is DriverState.DROWSY
        assert last.drowsy
        assert last.perclos_drowsy

    def test_perclos_triggers_the_drowsy_alarm_exactly_once(self):
        sm = DrowsinessStateMachine(
            make_settings(perclos_drowsy_threshold=50.0, perclos_drowsy_seconds=3.0), fps=30
        )
        triggers = 0
        for i in range(20):
            r = sm.update(0.30, 0.1, timestamp=i * 1.0, perclos=90.0)
            triggers += int(r.triggered)
        assert r.perclos_drowsy
        assert triggers == 1

    def test_perclos_returns_to_alert_after_drop(self):
        sm = DrowsinessStateMachine(
            make_settings(
                perclos_drowsy_threshold=60.0, perclos_drowsy_seconds=2.0, recovery_frames=5
            ),
            fps=30,
        )
        for i in range(30):
            sm.update(0.30, 0.1, timestamp=i * 0.1, perclos=90.0)
        assert sm.state is DriverState.DROWSY
        last = None
        for i in range(30, 100):
            last = sm.update(0.30, 0.1, timestamp=i * 0.1, perclos=5.0)
        assert last.state is DriverState.ALERT
        assert not last.perclos_drowsy
