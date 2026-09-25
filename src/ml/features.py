"""Windowed feature extraction for the optional ML classifier.

Every call appends the current (EAR, MAR) pair to a rolling window and returns a
fixed-length numeric vector. Classic "tells" of drowsiness that these features
capture:

  * mean EAR        - baseline of how open the eyes are,
  * min / q10 EAR   - how closed the eyes get (micro-sleeps),
  * EAR coefficient of variation - erratic eyelid behaviour,
  * MAR mean / max  - persistent mouth openness / yawns,
  * closure fraction - share of recent frames with EAR < threshold,
  * openness fraction - share of recent frames with MAR > threshold.

The extractor is pure NumPy so it works identically at training and inference.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray


class FeatureExtractor:
    """Maintains a rolling window of EAR/MAR and produces feature vectors."""

    def __init__(
        self,
        window: int = 45,
        ear_threshold: float = 0.25,
        mar_threshold: float = 0.55,
    ) -> None:
        self.window = max(2, window)
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self._ear = deque(maxlen=self.window)
        self._mar = deque(maxlen=self.window)

    # ------------------------------------------------------------- public ---
    def update(self, ear: float, mar: float) -> None:
        self._ear.append(float(ear))
        self._mar.append(float(mar))

    def reset(self) -> None:
        self._ear.clear()
        self._mar.clear()

    def update_and_features(self, ear: float, mar: float) -> NDArray[np.floating]:
        self.update(ear, mar)
        return self.features()

    def features(self) -> NDArray[np.floating]:
        """Numeric vector; padded with NaN until the window is warm (>=2)."""
        n = len(self._ear)
        if n < 2:
            return np.full(self.dimension(), np.nan, dtype=np.float32)

        ear = np.asarray(self._ear, dtype=np.float64)
        mar = np.asarray(self._mar, dtype=np.float64)

        def safe_std(a: NDArray[np.floating]) -> float:
            return float(np.std(a)) if a.size > 1 else 0.0

        vec = [
            float(np.mean(ear)),
            float(np.min(ear)),
            float(np.percentile(ear, 10)),
            safe_std(ear),
            float(np.mean(mar)),
            float(np.max(mar)),
            safe_std(mar),
            float(np.mean(ear < self.ear_threshold)),   # eye closed fraction
            float(np.mean(mar > self.mar_threshold)),   # mouth open fraction
            float(np.max(mar) - np.min(mar)),           # mouth dynamic range
        ]
        return np.asarray(vec, dtype=np.float32)

    @staticmethod
    def dimension() -> int:
        return 10

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "ear_mean",
            "ear_min",
            "ear_q10",
            "ear_std",
            "mar_mean",
            "mar_max",
            "mar_std",
            "eye_closed_fraction",
            "mouth_open_fraction",
            "mar_range",
        ]


def extract_batch_features(
    ear_seq: NDArray[np.floating],
    mar_seq: NDArray[np.floating],
    window: int = 45,
    ear_threshold: float = 0.25,
    mar_threshold: float = 0.55,
) -> NDArray[np.floating]:
    """Vectorised feature extraction over a whole sequence (training helper).

    Returns an (T, D) matrix; row ``t`` describes the window *ending* at ``t``.
    """
    ear_seq = np.asarray(ear_seq, dtype=np.float64)
    mar_seq = np.asarray(mar_seq, dtype=np.float64)
    assert ear_seq.shape == mar_seq.shape

    t_total = len(ear_seq)
    dim = 10
    out = np.full((t_total, dim), np.nan, dtype=np.float32)

    running_ear: list[float] = []
    running_mar: list[float] = []

    for t in range(t_total):
        running_ear.append(float(ear_seq[t]))
        running_mar.append(float(mar_seq[t]))
        if len(running_ear) > window:
            running_ear.pop(0)
            running_mar.pop(0)
        if len(running_ear) < window:
            continue  # window not warm -> stay NaN
        ear = np.asarray(running_ear)
        mar = np.asarray(running_mar)
        out[t] = [
            float(ear.mean()),
            float(ear.min()),
            float(np.percentile(ear, 10)),
            float(ear.std()),
            float(mar.mean()),
            float(mar.max()),
            float(mar.std()),
            float(np.mean(ear < ear_threshold)),
            float(np.mean(mar > mar_threshold)),
            float(mar.max() - mar.min()),
        ]
    return out


__all__ = ["FeatureExtractor", "extract_batch_features"]
