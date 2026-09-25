"""Alert coordinator: maps state-machine events to audio + visual actions.

Logic summary
------------
* A *confirmed* yawn (``report.yawn_triggered``) fires the SAME loud siren as
  the drowsiness alarm (``AudioAlert.start_siren``), rate-limited only by the
  alert cooldown. The very first yawn of a session is never blocked.
* DROWSY fired    -> loud looping siren + full-screen flash; the siren keeps
                     playing until the driver recovers (hysteresis handled by
                     the state machine) so a *prolonged* issue stays loud.
* Cooldown        -> separate triggers at least ``alert_cooldown_seconds`` apart
                     prevent the alarm from chattering on borderline signals.
"""

from __future__ import annotations

import logging
import time

from src.alerts.audio import AudioAlert
from src.config.settings import AlertSettings
from src.dynamics.state_machine import DriverState, FrameReport

logger = logging.getLogger(__name__)


class AlertManager:
    """Turns ``FrameReport`` events into audio/visual warnings."""

    def __init__(
        self,
        audio: AudioAlert | None = None,
        settings: AlertSettings | None = None,
    ) -> None:
        self.settings = settings or AlertSettings()
        self.audio = audio
        self._last_siren_at = 0.0
        self._siren_active = False
        self._last_yawn_alarm_at = 0.0   # 0.0 => first yawn is never blocked
        self._yawn_siren_until = 0.0
        self.drowsy_events: int = 0  # lifecycle counters exposed to the UI
        self.yawn_events: int = 0

    # --------------------------------------------------------------- audio --
    def _ensure_audio(self) -> AudioAlert | None:
        if self.audio is None and self.settings.audio_enabled:
            try:
                self.audio = AudioAlert(volume=self.settings.audio_volume)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Audio unavailable, running visual-only: %s", exc)
                self.audio = None
        return self.audio

    # -------------------------------------------------------------- update ---
    def update(self, report: FrameReport, now: float | None = None) -> bool:
        """React to a processed frame; returns True if a warning fired."""
        now = now if now is not None else time.monotonic()
        fired = False

        # -- DROWSY: fire the siren on the transition, keep it until recovery.
        if report.drowsy:
            if (
                not self._siren_active
                and now - self._last_siren_at >= self.settings.alert_cooldown_seconds
            ):
                audio = self._ensure_audio()
                if audio is not None:
                    audio.start_siren()
                else:
                    logger.critical("*** DROWSINESS ALERT ***")
                self._siren_active = True
                self._last_siren_at = now
                self.drowsy_events += 1
                fired = True
        elif self._siren_active and report.state is not DriverState.DROWSY:
            # Driver recovered -> silence the alarm.
            if self.audio is not None:
                self.audio.stop_siren()
            self._siren_active = False

        # -- YAWN: a confirmed yawn uses the SAME loud alarm as drowsiness, so
        # the driver is woken regardless of which trigger fired first.
        if (
            report.yawn_triggered
            and self.settings.yawn_alert_enabled
            and not report.drowsy
        ):
            # Cooldown prevents alarm spam between *separate* yawns, but the
            # first yawning alert of the session is always allowed through.
            if (
                self._last_yawn_alarm_at == 0.0
                or now - self._last_yawn_alarm_at >= self.settings.alert_cooldown_seconds
            ):
                self._last_yawn_alarm_at = now
                audio = self._ensure_audio()
                if audio is not None:
                    audio.start_siren()
                    self._yawn_siren_until = now + self.settings.yawn_siren_seconds
                    logger.warning("Triggering audio alert for yawn")
                else:
                    logger.critical("*** YAWN DETECTED (audio unavailable) ***")
                self.yawn_events += 1
                fired = True

        # A yawn siren is a short burst: stop it once its window has elapsed so
        # it never loops forever (and never muffles a real drowsiness alarm).
        if self._yawn_siren_until and now >= self._yawn_siren_until:
            if not self._siren_active and self.audio is not None:
                self.audio.stop_siren()
            self._yawn_siren_until = 0.0

        # -- face lost: if we see nothing for a long time, remind the driver.
        if (
            report.face_present is False
            and report.no_face_frames % 90 == 1
            and report.no_face_frames >= 2
        ):
            audio = self._ensure_audio()
            if audio is not None:
                audio.play_warning_beep(min_interval=2.0)
        return fired

    def close(self) -> None:
        if self.audio is not None:
            self.audio.close()


__all__ = ["AlertManager"]
