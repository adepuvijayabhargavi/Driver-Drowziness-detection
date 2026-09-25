"""Simple FPS meter with a smoothed moving average."""

from __future__ import annotations

import time


class FPSMeter:
    def __init__(self, smoothing: float = 0.9, report_every: int | None = None) -> None:
        self.smoothing = smoothing
        self.report_every = report_every
        self._last = time.perf_counter()
        self._ema: float | None = None
        self._frames = 0
        self._since_report = 0

    def tick(self) -> float:
        """Call once per rendered frame; returns the current smoothed FPS."""
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt <= 0:
            return self._ema or 0.0
        inst = 1.0 / dt
        self._ema = inst if self._ema is None else (
            self.smoothing * self._ema + (1.0 - self.smoothing) * inst
        )
        self._frames += 1
        return self._ema

    @property
    def mean_fps(self) -> float:
        return self._ema or 0.0


def sleep_to_fps(target_fps: float, frame_start: float) -> None:
    """Block until the per-frame budget for ``target_fps`` is spent."""
    import time

    if target_fps <= 0:
        return
    budget = 1.0 / target_fps
    elapsed = time.perf_counter() - frame_start
    remain = budget - elapsed
    if remain > 0:
        time.sleep(remain)


__all__ = ["FPSMeter", "sleep_to_fps"]
