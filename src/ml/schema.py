"""Single source of truth for the supervised ML feature schema.

Every ML artifact in the project -- the data-collection script, the trainer and
the real-time classifier -- agrees on ONE ordered list of features defined
below. Storing the order here (instead of in three separate places) is what
guarantees the same feature vector is produced at training time and at
inference time.

A dataset row looks like::

    timestamp, session_id, ear_left, ear_right, ear_mean, mar, perclos,
    eye_closed_duration, yawn_duration, pitch, yaw, roll, label

Rules
-----
* Features are ONLY the values the CV pipeline already computes per frame.
  Nothing is invented or duplicated here: ``ear_left/ear_right`` come from the
  per-eye ``eye_aspect_ratio``, ``ear_mean`` is the average EAR, ``perclos`` is
  the rolling PERCLOS percentage, ``eye_closed_duration`` / ``yawn_duration``
  are measured (seconds) by the state machine, and ``pitch/yaw/roll`` are the
  head-pose angles.
* ``None`` is a legitimate reading of an unavailable signal and is encoded as
  an invalid (NaN) value; it is never fabricated into a number.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Ordered feature columns. DO NOT reorder -- saved models embed this exact
# ordering and the classifier refuses to run against a mismatched schema.
FEATURE_COLUMNS: tuple[str, ...] = (
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

N_FEATURES: int = len(FEATURE_COLUMNS)

# Row metadata that surrounds the feature columns in a dataset CSV.
DIM_COLUMNS: tuple[str, ...] = ("timestamp", "session_id")
LABEL_COLUMN: str = "label"
# Valid labels; the collector/trainer/classifier all normalize to these.
LABELS: tuple[str, ...] = ("alert", "drowsy")

HEADER: tuple[str, ...] = DIM_COLUMNS + FEATURE_COLUMNS + (LABEL_COLUMN,)

# How each schema feature is read from a FrameReport attribute.
_REPORT_ATTRS: Mapping[str, str] = {
    "ear_left": "ear_left",
    "ear_right": "ear_right",
    "ear_mean": "ear",
    "mar": "mar",
    "perclos": "perclos",
    "eye_closed_duration": "eye_closed_duration",
    "yawn_duration": "yawn_duration",
    "pitch": "head_pitch",
    "yaw": "head_yaw",
    "roll": "head_roll",
}


def feature_names() -> list[str]:
    """Return the canonical feature list (preserving order)."""
    return list(FEATURE_COLUMNS)


def validate_feature_order(names: Sequence[str]) -> bool:
    """True only if ``names`` matches FEATURE_COLUMNS in the exact order.

    The classifier refuses to run against a model whose schema it cannot
    replicate, so inference can never silently feed features in the wrong order.
    """
    return [str(n) for n in names] == list(FEATURE_COLUMNS)


def feature_vector_from_report(report) -> np.ndarray:
    """Build the (N_FEATURES,) float32 vector for ``report``.

    Attributes that are ``None`` (signal unavailable for this frame) become
    NaN. Callers decide how to treat NaN rows (the collector skips the row
    only when the *core* face features are invalid; the trainer drops any row
    containing NaN).
    """
    values: list[float] = []
    for column in FEATURE_COLUMNS:
        value = getattr(report, _REPORT_ATTRS[column], None)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float32)


def feature_row(
    report,
    *,
    timestamp: float,
    session_id: object,
    label: str,
) -> dict[str, object]:
    """One dataset row for ``report`` ready to be written to a CSV.

    ``None`` / invalid numeric values are written as empty strings (the same
    convention the session logger already uses) and parsed back as NaN by
    :func:`src.ml.dataset.load_dataset`.
    """
    row: dict[str, object] = {
        "timestamp": float(timestamp),
        "session_id": str(session_id),
    }
    vector = feature_vector_from_report(report)
    for column, value in zip(FEATURE_COLUMNS, vector, strict=True):
        value_f = float(value)
        row[column] = "" if np.isnan(value_f) else f"{value_f:.5g}"
    if label not in LABELS:
        raise ValueError(
            f"invalid label {label!r}; expected one of {LABELS}"
        )
    row[LABEL_COLUMN] = label
    return row


__all__ = [
    "FEATURE_COLUMNS",
    "N_FEATURES",
    "DIM_COLUMNS",
    "LABEL_COLUMN",
    "LABELS",
    "HEADER",
    "feature_names",
    "validate_feature_order",
    "feature_vector_from_report",
    "feature_row",
]
