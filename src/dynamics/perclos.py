"""Rolling-window PERCLOS (Percentage of Eye Closure) calculator.

PERCLOS is the fraction of *time* -- over a trailing ``window_seconds`` -- in
which the driver's eyes are substantially closed, i.e. the Eye Aspect Ratio is
below ``ear_threshold``.

This module is deliberately **OpenCV/MediaPipe-free** (like the rest of
``src/dynamics``): it only consumes ``(ear, timestamp)`` pairs, so it is
trivial to unit test and reusable from any face-landmark backend.

Design notes
------------
* Time, not frame count: every sample is timestamped and elapsed time is
  derived from real timestamps, so a variable-FPS or dropped-frame camera does
  not distort the ratio.
* Intervals between *valid* samples are attributed to the eye state observed
  when the interval closes. A gap longer than ``max_gap_seconds`` (camera
  covered, the face left the frame, several seconds of dropped frames) is
  discarded -- that unobserved span is counted as neither eye-open nor
  eye-closed time.
* Invalid samples (missing, NaN, infinite or negative EAR) are ignored and
  therefore never counted as closed eyes.
* A ``deque`` stores only the finished intervals; entries are pruned in
  amortized O(1) once they leave the window.
"""

from __future__ import annotations

import math
from collections import deque

__all__ = ["PerclosCalculator"]


class PerclosCalculator:
    """Tracks closed-eye time and monitoring time inside a sliding time window."""

    def __init__(
        self,
        window_seconds: float = 60.0,
        ear_threshold: float = 0.25,
        max_gap_seconds: float = 1.0,
    ) -> None:
        self.window_seconds = max(float(window_seconds), 1e-3)
        self.ear_threshold = float(ear_threshold)
        self.max_gap_seconds = max(float(max_gap_seconds), 1e-3)
        self._intervals: deque[tuple[float, float, bool]] = deque()
        self._closed_time = 0.0
        self._monitoring_time = 0.0
        self._last: tuple[float, float] | None = None
        self._now = 0.0

    # -------------------------------------------------------------- public --
    def update(self, ear: float | None, timestamp: float) -> None:
        """Feed one measurement.

        ``ear=None`` (or any non-finite / negative value) records nothing but
        still advances the internal clock, so old samples keep expiring even
        while the face is out of frame.
        """
        self._now = max(self._now, float(timestamp))
        self._prune(self._now)

        if ear is None:
            return
        try:
            value = float(ear)
        except (TypeError, ValueError):
            return
        if not (math.isfinite(value) and value >= 0.0):
            return

        closed = value < self.ear_threshold
        if self._last is not None:
            interval = timestamp - self._last[0]
            if 0.0 < interval <= self.max_gap_seconds:
                self._intervals.append((timestamp, interval, closed))
                self._monitoring_time += interval
                if closed:
                    self._closed_time += interval
        self._last = (timestamp, value)

    def reset(self) -> None:
        """Clear all history (used when a session/run is restarted)."""
        self._intervals.clear()
        self._closed_time = 0.0
        self._monitoring_time = 0.0
        self._last = None
        self._now = 0.0

    # ----------------------------------------------------------- properties --
    @property
    def perclos(self) -> float:
        """Percentage of eye closure over the window; 0.0 if nothing yet."""
        total = self.monitoring_time
        if total <= 0.0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * self.closed_time / total))

    @property
    def monitoring_time(self) -> float:
        """Seconds of valid monitoring inside the window (capped intervals)."""
        self._prune(self._now)
        return max(0.0, self._monitoring_time + self._tail_time())

    @property
    def closed_time(self) -> float:
        """Seconds counted as eyes-closed inside the window."""
        self._prune(self._now)
        return max(0.0, self._closed_time + self._tail_closed_time())

    @property
    def sample_count(self) -> int:
        """Number of valid measurements still represented in the window."""
        self._prune(self._now)
        count = len(self._intervals)
        if self._last is not None and self._last[0] > self._now - self.window_seconds:
            count += 1
        return count

    # ------------------------------------------------------------ internals --
    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._intervals and self._intervals[0][0] <= cutoff:
            _, duration, closed = self._intervals.popleft()
            self._monitoring_time -= duration
            if closed:
                self._closed_time -= duration

    def _tail_time(self) -> float:
        """Duration of the not-yet-closed interval since the last valid sample."""
        if self._last is None:
            return 0.0
        tail = self._now - self._last[0]
        if 0.0 < tail <= self.max_gap_seconds:
            return tail
        return 0.0

    def _tail_closed_time(self) -> float:
        tail = self._tail_time()
        if tail == 0.0 or self._last is None:
            return 0.0
        return tail if self._last[1] < self.ear_threshold else 0.0
