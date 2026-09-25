"""End-to-end pipeline tests using a synthetic detector (no webcam needed)."""

from __future__ import annotations

import numpy as np
import pytest

from src.alerts.audio import synthesize_alert_wave
from src.config.settings import Settings
from src.dynamics.state_machine import DriverState
from src.pipeline import DrowsinessPipeline
from src.vision.detector import SyntheticFaceDetector
from tests.conftest import face_with_ear, face_with_mar


@pytest.fixture
def settings():
    return Settings().with_detection(
        ear_consecutive_frames=15,
        mar_consecutive_frames=15,
        recovery_frames=5,
    ).with_alert(audio_enabled=False)


@pytest.fixture
def pipeline(settings):
    return DrowsinessPipeline(
        detector=SyntheticFaceDetector(face_with_ear(0.05)),  # closed eyes
        settings=settings,
    )


def blank_frame(shape=(320, 240, 3)) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


class TestPipelineSyntheticFaces:
    def test_open_face_stays_alert(self, settings):
        pl = DrowsinessPipeline(
            detector=SyntheticFaceDetector(face_with_ear(0.3)),
            settings=settings,
        )
        last = None
        for _ in range(50):
            annotated, report = pl.process_frame(blank_frame())
            last = report
        assert last.state is DriverState.ALERT
        assert not last.drowsy
        assert annotated.shape == blank_frame().shape

    def test_closed_eyes_trigger_drowsy(self, pipeline):
        last = None
        triggers = 0
        for _ in range(40):
            annotated, report = pipeline.process_frame(blank_frame())
            last = report
            triggers += int(report.triggered)
        assert last.drowsy
        assert triggers == 1
        assert annotated is not None

    def test_out_of_bounds_frame_size_ok(self, pipeline):
        annotated_small = pipeline.process_frame(np.zeros((120, 160, 3), dtype=np.uint8))[0]
        assert annotated_small.shape == (120, 160, 3)

    def test_annotated_frame_is_not_modified_input(self, pipeline):
        frame = np.full((240, 320, 3), 120, dtype=np.uint8)
        original = frame.copy()
        pipeline.process_frame(frame)
        assert np.array_equal(frame, original)

    def test_yawn_pipeline_detects_yawning(self, settings):
        pl = DrowsinessPipeline(
            detector=SyntheticFaceDetector(face_with_mar(0.8)),
            settings=settings,
        )
        last = None
        for _ in range(20):
            _, report = pl.process_frame(blank_frame())
            last = report
        assert last.yawning
        assert last.state in (DriverState.YAWNING, DriverState.ALERT)


class TestAudioGeneration:
    def test_synthesize_writes_wave(self, tmp_path):
        out = tmp_path / "siren.wav"
        synthesize_alert_wave(out, duration=0.2)
        assert out.exists()
        assert out.stat().st_size > 1000


class TestAlertManagerCooldown:
    def _manager_and_state(self, settings):
        from src.alerts.manager import AlertManager
        from src.config.settings import AlertSettings
        from src.dynamics.state_machine import DrowsinessStateMachine

        alert_settings = AlertSettings(audio_enabled=False)
        m = AlertManager(settings=alert_settings)
        sm = DrowsinessStateMachine(settings.detection, fps=30)
        return m, sm

    def _drive_drowsy(self, sm):
        for _ in range(20):
            r = sm.update(0.05, 0.1)
        return r

    def test_drowsy_events_incremented(self, settings):
        m, sm = self._manager_and_state(settings)
        report = self._drive_drowsy(sm)
        assert m.update(report, now=10.0) is True
        assert m.drowsy_events == 1

    def test_cooldown_blocks_repeat_trigger(self, settings):
        m, sm = self._manager_and_state(settings)

        # First trigger fires at t=10.
        report = self._drive_drowsy(sm)
        assert m.update(report, now=10.0) is True
        assert m.drowsy_events == 1

        # The driver wakes up; the manager observes recovery and stops the siren.
        for _ in range(20):
            m.update(sm.update(0.35, 0.1), now=11.0)
        assert m._siren_active is False

        # Falling asleep again just 2 s later: cooldown blocks the repeat.
        sm.reset()
        report2 = self._drive_drowsy(sm)
        assert m.update(report2, now=12.0) is False
        assert m.drowsy_events == 1

        # After 20 s (>5 s cooldown) a new drowsy spell may alarm again.
        report3 = self._drive_drowsy(sm)
        assert m.update(report3, now=30.0) is True
        assert m.drowsy_events == 2

    def test_yawn_alert_triggers_same_siren_as_drowsy(self, settings):
        from src.alerts.manager import AlertManager
        from src.config.settings import AlertSettings
        from src.dynamics.state_machine import DrowsinessStateMachine

        call_log: dict[str, int] = {"sirens": 0, "stops": 0}

        class FakeAudio:
            def start_siren(self) -> None:
                call_log["sirens"] += 1

            def stop_siren(self) -> None:
                call_log["stops"] += 1

            def close(self) -> None:
                pass

        m = AlertManager(
            settings=AlertSettings(yawn_alert_enabled=True, alert_cooldown_seconds=5.0)
        )
        sm = DrowsinessStateMachine(settings.detection, fps=30)
        m.audio = FakeAudio()   # inject the fake so no pygame is initialised

        report = None
        for _ in range(20):
            r = sm.update(0.3, 0.9)   # sustained mouth open -> confirmed
            if r.yawn_triggered:
                report = r            # the exact frame the yawn was confirmed
        assert report is not None and report.yawn_triggered
        fired = m.update(report, now=5.0)
        assert fired is True
        assert m.yawn_events == 1
        assert call_log["sirens"] == 1     # the SAME start_siren as drowsiness
        # The yawn siren is a short burst that auto-stops once its window ends.
        m.update(report, now=5.0 + 2.1)
        assert call_log["stops"] == 1

    def test_yawn_first_alert_never_blocked_then_cooldown_between_yawns(self, settings):
        from src.alerts.manager import AlertManager
        from src.config.settings import AlertSettings
        from src.dynamics.state_machine import DrowsinessStateMachine

        call_log: dict[str, int] = {"sirens": 0}

        class FakeAudio:
            def start_siren(self) -> None:
                call_log["sirens"] += 1

            def stop_siren(self) -> None:
                pass

            def close(self) -> None:
                pass

        m = AlertManager(
            settings=AlertSettings(yawn_alert_enabled=True, alert_cooldown_seconds=5.0)
        )
        m.audio = FakeAudio()

        def confirmed_yawn_report():
            sm = DrowsinessStateMachine(settings.detection, fps=30)
            for _ in range(20):
                r = sm.update(0.3, 0.9)
                if r.yawn_triggered:
                    return r
            raise AssertionError("Yawn was not confirmed in 20 frames")

        # The very first yawn alert fires (never blocked by any earlier event).
        assert m.update(confirmed_yawn_report(), now=2.0) is True
        assert m.yawn_events == 1
        assert call_log["sirens"] == 1

        # A second yawn inside the 5 s cooldown is suppressed.
        assert m.update(confirmed_yawn_report(), now=3.0) is False
        assert call_log["sirens"] == 1

        # Once the cooldown has passed a new yawn may alarm again.
        assert m.update(confirmed_yawn_report(), now=8.0) is True
        assert m.yawn_events == 2
        assert call_log["sirens"] == 2
