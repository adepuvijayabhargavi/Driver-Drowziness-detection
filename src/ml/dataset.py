"""Dataset I/O and validation for the supervised ML pipeline.

Pure pandas-free helpers that load the CSVs written by
:mod:`scripts.collect_ml_data` into numpy arrays, validate them, drop rows
that cannot be used for training (missing/invalid measurements, empty labels)
and split sessions into train/test WITHOUT data leakage.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.ml.schema import (
    DIM_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    LABELS,
)

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = (LABEL_COLUMN,) + DIM_COLUMNS + FEATURE_COLUMNS


class DatasetError(ValueError):
    """Raised when a dataset is malformed (missing columns, bad labels...)."""


@dataclass
class Dataset:
    """Validated training data.

    ``features`` has shape (N, N_FEATURES) aligned with FEATURE_COLUMNS;
    ``labels`` holds raw alert/drowsy strings and ``sessions`` the session ids.
    """

    features: NDArray[np.floating]
    labels: NDArray[np.str_]
    sessions: NDArray[np.str_]
    timestamps: NDArray[np.floating]
    sources: list[str]

    @property
    def size(self) -> int:
        return int(self.features.shape[0])


def _parse_float(value: str) -> float:
    text = value.strip()
    if text == "":
        return float("nan")
    try:
        result = float(text)
    except ValueError:
        return float("nan")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"{path}: empty CSV file")
        missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise DatasetError(f"{path}: missing required column(s): {missing}")
        return list(reader)


def load_dataset(paths: str | Path | Sequence[str | Path]) -> Dataset:
    """Load and merge one or more dataset CSV files into a :class:`Dataset`."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    sources: list[str] = []
    features: list[np.ndarray] = []
    labels: list[str] = []
    sessions: list[str] = []
    timestamps: list[float] = []

    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise DatasetError(f"dataset file not found: {path}")
        rows = _read_csv(path)
        n_bad_labels = 0
        for row in rows:
            label = row[LABEL_COLUMN].strip().lower()
            if label not in LABELS:
                n_bad_labels += 1
                continue
            features.append(
                np.asarray(
                    [_parse_float(row[col]) for col in FEATURE_COLUMNS],
                    dtype=np.float64,
                )
            )
            labels.append(label)
            sessions.append(row["session_id"].strip())
            timestamps.append(_parse_float(row["timestamp"]))
        if n_bad_labels:
            logger.warning(
                "%s: skipped %d row(s) with invalid labels", path, n_bad_labels
            )
        sources.append(str(path))

    if not features:
        raise DatasetError(
            "no usable rows found; check the CSV layout and labels"
        )

    return Dataset(
        features=np.asarray(features, dtype=np.float64),
        labels=np.asarray(labels, dtype=str),
        sessions=np.asarray(sessions, dtype=str),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        sources=sources,
    )


def drop_invalid_rows(
    dataset: Dataset,
) -> tuple[Dataset, int]:
    """Remove rows that cannot train or evaluate a classifier.

    Drops rows where any feature is NaN (an unavailable measurement), or where
    the session id is empty (such a row cannot be safely grouped). Returns the
    cleaned dataset and the number of removed rows.
    """
    finite = np.isfinite(dataset.features).all(axis=1)
    has_session = dataset.sessions != ""
    keep = finite & has_session
    dropped = int(keep.size - int(np.count_nonzero(keep)))
    if dropped:
        logger.info("dropped %d invalid row(s) (NaN features / empty session)", dropped)
    return Dataset(
        features=dataset.features[keep],
        labels=dataset.labels[keep],
        sessions=dataset.sessions[keep],
        timestamps=dataset.timestamps[keep],
        sources=dataset.sources,
    ), dropped


def describe_missing(dataset: Dataset) -> dict[str, object]:
    """Count missing/invalid values per feature column.

    A value is "missing/invalid" when it is not a finite number: empty cells,
    NaN (unavailable measurement) and infinities. Also counts rows whose
    ``session_id`` is empty (those cannot be safely grouped). Used to report
    exactly what will be dropped before training.
    """
    columns: dict[str, int] = {}
    for index, column in enumerate(FEATURE_COLUMNS):
        columns[column] = int(
            np.count_nonzero(~np.isfinite(dataset.features[:, index]))
        )
    return {
        "columns": columns,
        "empty_session": int(np.count_nonzero(dataset.sessions == "")),
    }


def validate_dataset(dataset: Dataset) -> list[str]:
    """Return a list of human-readable issues with ``dataset``.

    An empty list means the dataset is usable. Issues are warnings about
    distribution and size -- the caller decides whether they are fatal.
    """
    issues: list[str] = []
    if dataset.size == 0:
        issues.append("dataset is empty")
        return issues
    if np.isnan(dataset.features).any():
        issues.append("dataset still contains NaN feature values")
    unique_labels = set(dataset.labels)
    missing = [lab for lab in LABELS if lab not in unique_labels]
    if missing:
        issues.append(f"no samples for label(s): {missing}")
    counts = {lab: int(np.count_nonzero(dataset.labels == lab)) for lab in LABELS}
    for lab, count in counts.items():
        issues.append(f"{count} {lab} sample(s)")
    if dataset.size < 20:
        issues.append(
            "dataset is very small (<20 rows); any trained model will not "
            "generalize - treat results as exploratory only"
        )
    n_sessions = len(set(dataset.sessions))
    if n_sessions < 2:
        issues.append(
            "only one session in the dataset; a session split cannot "
            "measure generalization"
        )
    issues.append(f"{n_sessions} session(s)")
    return issues


def group_train_test_split(
    dataset: Dataset,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows by WHOLE session (no data leakage).

    The resulting test rows come exclusively from sessions absent from the
    training rows -- a model can never "remember" a driver's session during
    evaluation. Returns (train_idx, test_idx) index arrays.

    Falls back to a plain randomized split only if scikit-learn is unavailable,
    with a loud warning (leakage risk).
    """
    try:
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError:  # pragma: no cover
        logger.warning(
            "scikit-learn is not installed; falling back to a random split "
            "(SESSION LEAKAGE RISK: sessions may appear in both splits)"
        )
        import numpy as _np

        perm = _np.random.RandomState(random_state).permutation(dataset.size)
        cut = int(round((1.0 - test_size) * dataset.size))
        return perm[:cut], perm[cut:]

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(
        splitter.split(dataset.features, dataset.labels, groups=dataset.sessions)
    )
    return train_idx, test_idx


if __name__ == "__main__":
    raise SystemExit("module is not meant to be run directly")

__all__ = [
    "Dataset",
    "DatasetError",
    "load_dataset",
    "drop_invalid_rows",
    "describe_missing",
    "validate_dataset",
    "group_train_test_split",
]
