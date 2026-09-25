"""Audio alert system.

Backend strategy (cross-platform, most-first):

1. ``pygame.mixer``   - primary: portable, low-latency, no extra DLLs.
2. ``winsound``       - Windows fallback (native, zero dependencies).
3. terminal bell      - last resort (asynchronous ``\a``).

If no sound backend initializes, the system keeps running silently and logs a
warning -- drowsiness detection must never crash because audio is unavailable.
"""

from __future__ import annotations

import logging
import math
import time
import wave
from pathlib import Path

import numpy as np

from src.config.settings import ASSETS_DIR, DEFAULT_ALERT_SOUND_PATH

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100


def synthesize_alert_wave(
    path: str | Path,
    duration: float = 0.60,
    freq_low: float = 880.0,
    freq_high: float = 1318.0,
    cycle: float = 0.25,
    amplitude: float = 0.85,
    sample_rate: int = SAMPLE_RATE,
    harmonics: int = 4,
) -> Path:
    """Write a loud two-tone 'siren' WAV file (built with NumPy only).

    The siren alternates between ``freq_low``/``freq_high`` every ``cycle``
    seconds and mixes several harmonics so it cuts through engine noise. This
    file is generated once on first run; nothing ships as a binary asset.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    # Phase block that switches frequency every `cycle` seconds.
    tone = (t // cycle) % 2
    freq = np.where(tone == 0, freq_low, freq_high)
    phase = 2.0 * math.pi * np.cumsum(freq) / sample_rate

    signal = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        signal += (1.0 / h) * np.sin(h * phase)
    signal /= harmonics
    signal *= amplitude

    pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    logger.debug("Alert wav written to %s", path)
    return path


class AudioAlert:
    """Play the drowsiness siren and short warning beeps."""

    def __init__(
        self,
        siren_file: str | Path | None = None,
        volume: float = 0.9,
    ) -> None:
        self.siren_file = Path(siren_file) if siren_file else DEFAULT_ALERT_SOUND_PATH
        self.volume = max(0.0, min(1.0, volume))

        self._backend = "none"
        self._sound = None
        self._beep_sound = None
        self._siren_playing = False
        self._last_beep_at = 0.0

        if not self.siren_file.exists():
            synthesize_alert_wave(self.siren_file)
        self._init_pygame()
        if self._backend == "none":
            self._init_winsound()

        logger.info("Audio backend: %s", self._backend)

    # ------------------------------------------------------------ backends --
    def _init_pygame(self) -> None:
        try:
            import pygame

            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1)
            self._sound = pygame.mixer.Sound(str(self.siren_file))
            self._sound.set_volume(self.volume)
            self._beep_sound = pygame.mixer.Sound(str(self.siren_file))
            self._beep_sound.set_volume(max(0.2, self.volume * 0.6))
            self._backend = "pygame"
            self._pygame = pygame
        except Exception as exc:  # noqa: BLE001 - hardware/init can fail any way
            logger.debug("pygame mixer unavailable: %s", exc)

    def _init_winsound(self) -> None:
        try:
            import winsound  # Windows only

            winsound.PlaySound(str(self.siren_file), winsound.SND_FILENAME | winsound.SND_ASYNC)
            winsound.PlaySound(None, winsound.SND_PURGE)  # stop right away
            self._backend = "winsound"
        except ImportError:
            logger.debug("winsound unavailable (not on Windows?)")

    # --------------------------------------------------------------- public --
    @property
    def backend(self) -> str:
        return self._backend

    def start_siren(self) -> None:
        """Begin looping the loud siren (idempotent)."""
        if self._siren_playing or self._backend == "none":
            return
        if self._backend == "pygame":
            self._sound.play(loops=-1)
        else:
            import winsound

            winsound.PlaySound(
                str(self.siren_file),
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
            )
        self._siren_playing = True
        logger.warning("SIREN STARTED")

    def stop_siren(self) -> None:
        """Stop the looping siren (idempotent)."""
        if not self._siren_playing:
            return
        if self._backend == "pygame":
            self._sound.stop()
        else:
            try:
                import winsound

                winsound.PlaySound(None, winsound.SND_PURGE)
            except ImportError:
                pass
        self._siren_playing = False
        logger.warning("SIREN STOPPED")

    def play_warning_beep(self, min_interval: float = 1.5) -> bool:
        """Play a short warning beep (face-lost nudge). Rate-limited; True if played."""
        now = time.monotonic()
        if now - self._last_beep_at < min_interval:
            return False
        self._last_beep_at = now
        if self._backend == "pygame":
            self._beep_sound.play()
            return True
        if self._backend == "winsound":
            import winsound

            winsound.Beep(880, 250)
            return True
        return False

    def close(self) -> None:
        self.stop_siren()
        if self._backend == "pygame":
            self._pygame.mixer.quit()


def make_assets_dir() -> Path:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    return ASSETS_DIR


__all__ = ["AudioAlert", "synthesize_alert_wave", "SAMPLE_RATE"]
