"""Tests for the temporal drowsiness state machine."""

from __future__ import annotations

from src.config.settings import DetectionSettings
from src.dynamics.state_machine import DriverState, DrowsinessStateMachine
from tests.conftest import EAR_THRESHOLD, MAR_THRESHOLD


def make_settings(**over) -> DetectionSettings:
    params = dict(
        ear_threshold=EAR_THRESHOLD,
        mar_threshold=MAR_THRESHOLD,
        ear_consecutive_frames=30,
        mar_consecutive_frames=30,
        min_blink_frames=2,
        recovery_frames=10,
    )
    params.update(over)
    return DetectionSettings(**params)


class TestEyes:
    def test_no_drowsiness_before_threshold(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        for _ in range(29):
            r = sm.update(0.05, 0.1)
            assert not r.drowsy
            assert not r.triggered
        assert sm.eye_closed_frames == 29

    def test_drowsiness_after_threshold_exactly(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        report = None
        for _ in range(30):
            report = sm.update(0.05, 0.1)
        assert report is not None
        assert report.drowsy
        assert report.triggered          # fired exactly on the boundary
        assert report.state is DriverState.DROWSY

    def test_triggered_only_once_while_still_closed(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        triggers = 0
        for _ in range(90):  # 3 s of closed eyes
            r = sm.update(0.05, 0.1)
            triggers += int(r.triggered)
        assert triggers == 1

    def test_recovery_requires_hysteresis(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        for _ in range(30):
            sm.update(0.05, 0.1)          # go DROWSY
        # Recover partially; must NOT leave DROWSY before recovery_frames.
        for _ in range(9):
            r = sm.update(0.35, 0.1)
            assert r.state is DriverState.DROWSY
            assert r.drowsy
        r = sm.update(0.35, 0.1)          # 10th open frame -> ALERT
        assert r.state is DriverState.ALERT
        assert not r.drowsy

    def test_single_short_closure_is_a_blink_not_drowsy(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        for _ in range(5):
            sm.update(0.05, 0.1)          # ~5-frame blink
        r = sm.update(0.35, 0.1)          # reopen
        assert r.blink_count == 1
        assert not r.drowsy
        assert r.state is DriverState.ALERT

    def test_no_face_freezes_counters(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        for _ in range(10):
            sm.update(0.05, 0.1)
        held = sm.eye_closed_frames
        r = sm.update(None, None)
        assert r.state is DriverState.UNKNOWN
        assert sm.eye_closed_frames == held  # not reset, not incremented
        assert r.no_face_frames == 1

    def test_blink_rate_computed(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        for _ in range(10):
            sm.update(0.35, 0.1)
        # 20 blinks, each a 2-frame closure followed by reopening.
        for _ in range(20):
            sm.update(0.05, 0.1)
            sm.update(0.05, 0.1)
            r = sm.update(0.35, 0.1)
        assert sm.blink_count == 20
        assert r.blink_rate_per_min > 0

    def test_min_blink_filter(self):
        sm = DrowsinessStateMachine(make_settings(min_blink_frames=5), fps=30)
        sm.update(0.05, 0.1)               # 1-frame micro-flutter
        r = sm.update(0.35, 0.1)
        assert r.blink_count == 0


class TestMouth:
    def test_yawn_detected_after_threshold(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        report = None
        for _ in range(30):
            report = sm.update(0.3, 0.8)   # mouth wide open
        assert report.state is DriverState.YAWNING
        assert report.yawning

    def test_yawn_not_fired_from_single_frame(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        r = sm.update(0.3, 0.9)
        assert r.state is DriverState.ALERT
        assert not r.yawning
        assert r.yawn_count == 0

    def test_yawn_count_incremented_on_completion(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        for _ in range(35):
            sm.update(0.3, 0.8)            # open mouth -> YAWNING, held
        r = sm.update(0.3, 0.1)            # mouth closes again
        assert r.yawn_count == 1
        assert not r.yawning

    def test_talking_short_opens_are_not_yawns(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        for _ in range(5):
            r = sm.update(0.3, 0.7)        # brief speech-like open
        r = sm.update(0.3, 0.1)
        assert r.yawn_count == 0
        assert r.state is DriverState.ALERT

    def test_yawn_confirmed_by_duration_at_low_fps(self):
        # 100 frames would need 100s at 10 fps -- only the duration bound (0.5s)
        # can trip. This proves detection is tied to real time, not to FPS.
        sm = DrowsinessStateMachine(
            make_settings(mar_consecutive_frames=100, yawn_duration_seconds=0.5),
            fps=10,
        )
        triggered = [False] * 10
        for i in range(10):
            r = sm.update(0.3, 0.85, timestamp=i * 0.1)
            triggered[i] = r.yawn_triggered
        assert r.yawning
        assert r.state is DriverState.YAWNING
        assert sum(triggered) == 1
        assert triggered[5] is True      # t=0.5s -> duration bound trips

    def test_yawn_triggered_only_once_while_held(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        triggers = 0
        for _ in range(60):
            r = sm.update(0.3, 0.8)
            triggers += int(r.yawn_triggered)
        assert triggers == 1
        assert r.yawning
        assert r.state is DriverState.YAWNING

    def test_yawn_counter_resets_only_when_mar_falls_below_threshold(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        for _ in range(10):
            r = sm.update(0.3, 0.8)
        assert r.yawn_frames == 10
        r = sm.update(0.3, 0.1)          # mouth closes -> counter resets
        assert r.yawn_frames == 0
        assert not r.yawning
        for _ in range(30):
            r = sm.update(0.3, 0.8)      # a fresh, independent yawn
        assert r.yawn_frames == 30
        assert r.yawning

    def test_eye_and_yawn_counters_are_independent(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        for _ in range(30):
            r = sm.update(0.35, 0.8)     # eyes open the whole time
        assert r.state is DriverState.YAWNING
        assert r.eye_closed_frames == 0
        assert r.yawn_frames == 30


class TestCombined:
    def test_drowsy_priority_over_yawn(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        for _ in range(30):
            r = sm.update(0.35, 0.8)       # 30 frames mouth open only
        assert r.state is DriverState.YAWNING
        sm.reset()
        for _ in range(30):
            r = sm.update(0.05, 0.8)       # eyes closed + mouth open
        assert r.state is DriverState.DROWSY

    def test_reset_clears_counters(self):
        sm = DrowsinessStateMachine(make_settings(), fps=30)
        for _ in range(30):
            sm.update(0.05, 0.1)
        assert sm.state is DriverState.DROWSY
        sm.reset()
        r = sm.update(0.35, 0.1)
        assert r.frame_index == 1
        assert sm.eye_closed_frames == 0
        assert sm.state is DriverState.ALERT
