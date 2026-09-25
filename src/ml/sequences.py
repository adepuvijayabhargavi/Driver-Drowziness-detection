"""Temporal sequence building for the deep-learning drowsiness classifier.

The DL layer learns from *sequences* of ``sequence_length`` consecutive frame
feature vectors instead of single frames. Two invariants matter here:

1. **A sequence never crosses a session boundary.** Samples are grouped by
   ``session_id`` and ordered chronologically by ``timestamp``; every window
   comes entirely from one session.
2. **Splitting happens BEFORE sequence generation.** Sessions are assigned to
   train / validation / test first, then sequences are built per fold -- so an
   overlapping window can never leak a session into two folds. (Overlap
   *inside* a session is intentional and harmless: sliding windows share frames
   only within the same session, which belongs to one fold.)

Labels: each collected session is labelled as a whole (``alert`` or ``drowsy``),
so every window inside a session has a uniform label. By default the window's
label is the **final frame's label** (``label_mode="final"``), which for this
dataset equals the majority (and every) label of the window.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.ml.dataset import Dataset, DatasetError
from src.ml.schema import LABELS

logger = logging.getLogger(__name__)

# Accepted label modes for a temporal window.
LABEL_MODES = ("final", "majority")


@dataclass(frozen=True)
class _SessionGroup:
    """Chronologically sorted row range of one session."""

    session_id: str
    label: str
    indices: NDArray[np.int_]  # row positions inside the Dataset


def group_by_session(dataset: Dataset, session_ids: list[str]) -> list[_SessionGroup]:
    """Return one :class:`_SessionGroup` per requested session, rows sorted by
    ``timestamp``. A requested session with no rows is skipped."""
    allowed = set(session_ids)
    ordered: dict[str, list[tuple[float, int]]] = {}
    labels: dict[str, str] = {}
    for index in range(dataset.size):
        session = str(dataset.sessions[index])
        if session not in allowed:
            continue
        ordered.setdefault(session, []).append(
            (float(dataset.timestamps[index]), index)
        )
        labels[session] = str(dataset.labels[index])
    groups: list[_SessionGroup] = []
    for session_id in session_ids:
        rows = ordered.get(session_id)
        if not rows:
            continue
        rows.sort(key=lambda pair: pair[0])
        groups.append(
            _SessionGroup(
                session_id=session_id,
                label=labels[session_id],
                indices=np.asarray([idx for _, idx in rows], dtype=np.int_),
            )
        )
    return groups


def _window_label(
    row_labels: NDArray[np.str_], mode: str
) -> tuple[str, bool]:
    """Resolve a window's label. Returns (label, mixed) where ``mixed`` is True
    only if the window contained more than one distinct label (should never
    happen with our session protocol)."""
    unique = np.unique(row_labels)
    if len(unique) == 1:
        return str(unique[0]), False
    label = str(unique[-1]) if mode == "final" else str(
        max(set(row_labels), key=list(row_labels).count)
    )
    logger.warning(
        "mixed-label window resolved to %r (%s label)", label, mode
    )
    return label, True


def build_sequences(
    dataset: Dataset,
    session_ids: list[str],
    *,
    sequence_length: int = 30,
    step: int = 1,
    label_mode: str = "final",
) -> tuple[NDArray[np.floating], NDArray[np.int_], dict[str, int]]:
    """Build fixed-length sliding windows from the given *whole sessions*.

    Parameters
    ----------
    dataset : Dataset
        Validated features/labels/sessions/timestamps.
    session_ids : list[str]
        Sessions that may contribute sequences (one fold only -- the caller is
        responsible for keeping folds disjoint).
    sequence_length : int
        Number of consecutive frames per window. Sessions shorter than this
        contribute no sequences (reported in the returned stats).
    step : int
        Sliding-window step (1 = fully overlapping windows).
    label_mode : str
        ``"final"`` uses the last frame's label; ``"majority"`` the majority.

    Returns
    -------
    (X, y, stats) with X shape (N, sequence_length, N_FEATURES), y the numeric
    labels (alert=0, drowsy=1) and a small stats dict.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}")

    blocks: list[NDArray[np.floating]] = []
    targets: list[int] = []
    skipped: list[str] = []
    mixed = 0
    for group in group_by_session(dataset, session_ids):
        rows = group.indices
        if len(rows) < sequence_length:
            skipped.append(f"{group.session_id} ({len(rows)} frames)")
            continue
        features = dataset.features[rows]
        row_labels = dataset.labels[rows]
        for start in range(0, len(rows) - sequence_length + 1, step):
            window = features[start : start + sequence_length]
            label, is_mixed = _window_label(
                row_labels[start : start + sequence_length], label_mode
            )
            mixed += int(is_mixed)
            blocks.append(window)
            targets.append(0 if label == "alert" else 1)
    if not blocks:
        raise DatasetError(
            f"no sequences could be built (sequence_length={sequence_length}); "
            f"sessions too short: {skipped or 'none provided'}"
        )
    stats: dict[str, int] = {
        "n_sequences": len(blocks),
        "sequence_length": sequence_length,
        "n_sequences_alert": sum(t == 0 for t in targets),
        "n_sequences_drowsy": sum(t == 1 for t in targets),
        "sessions_skipped_short": len(skipped),
        "mixed_label_windows": mixed,
    }
    if skipped:
        logger.info(
            "skipped %d session(s) shorter than %d frames: %s",
            len(skipped), sequence_length, skipped,
        )
    return (
        np.asarray(blocks, dtype=np.float32),
        np.asarray(targets, dtype=np.int_),
        stats,
    )


def split_sessions(
    dataset: Dataset,
    *,
    test_per_class: int = 1,
    val_per_class: int = 1,
    random_state: int = 42,
) -> dict[str, list[str]]:
    """Split session ids into train/validation/test, balanced per class.

    Sessions are the atomic unit -- a whole session may only ever end up in ONE
    fold, so not a single row leaks between folds. Per class the sessions are
    first shuffled with a seeded RNG (tie-breaking) and then re-ordered by
    *descending* row count so the LARGEST sessions contribute to TRAINING while
    the smallest ones are held out. Since periodic short sessions produce few
    windows, a plain random shuffle can accidentally starve training of an
    entire class (e.g. giving the only large ALERT session to validation).
    Ordering by size avoids that pathological outcome without weakening the
    session-based leakage prevention.

    Raises :class:`DatasetError` if a fold ends up with no session of either
    class (a single-class fold cannot train/validate an LSTM).
    """

    def _row_counts() -> dict[str, int]:
        counts: dict[str, int] = {}
        for index in range(dataset.size):
            session = str(dataset.sessions[index])
            counts[session] = counts.get(session, 0) + 1
        return counts

    row_counts = _row_counts()

    by_class: dict[str, list[str]] = {}
    for index in range(dataset.size):
        label = str(dataset.labels[index])
        session = str(dataset.sessions[index])
        if label not in by_class:
            by_class[label] = []
        if session not in by_class[label]:
            by_class[label].append(session)
    missing = [lab for lab in LABELS if not by_class.get(lab)]
    if missing:
        raise DatasetError(f"no sessions for label(s) {missing}; need both classes")

    rng = np.random.RandomState(random_state)
    folds: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for label, sessions in by_class.items():
        shuffled = list(sessions)
        rng.shuffle(shuffled)
        if len(shuffled) < test_per_class + val_per_class + 1:
            raise DatasetError(
                f"class {label!r} has {len(shuffled)} session(s), need at least "
                f"{test_per_class + val_per_class + 1} for test/val/train"
            )
        # Stable sort: keeps the seeded shuffle order for equal-sized sessions,
        # but guarantees the largest sessions always end up in TRAINING by
        # placing the few smallest in test/validation and the rest in train.
        ordered = sorted(
            shuffled,
            key=lambda session: row_counts.get(session, 0),
        )
        folders = ["test", "validation"]
        counts = [test_per_class, val_per_class]
        lo = 0
        for folder, count in zip(folders, counts):
            folds[folder].extend(ordered[lo : lo + count])
            lo += count
        folds["train"].extend(ordered[lo:])

    for fold, sessions in folds.items():
        labels = {
            str(dataset.labels[i])
            for i in range(dataset.size)
            if str(dataset.sessions[i]) in sessions
        }
        if labels < set(LABELS):
            raise DatasetError(
                f"the {fold} fold only covers label(s) {labels}; pick different "
                "test/val session counts or sequence length so every fold has "
                "both classes"
            )
    return folds


def fit_scaler(X_train: NDArray[np.floating]):
    """Fit a StandardScaler on the TRAINING sequences only.

    The scaler is fit over every frame of the training sequences (flattened) so
    per-feature statistics never see validation/test data.
    """
    from sklearn.preprocessing import StandardScaler

    flat = X_train.reshape(-1, X_train.shape[-1])
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def scale_sequences(
    X: NDArray[np.floating], scaler,
) -> NDArray[np.floating]:
    """Apply the fitted scaler to (N, seq_len, n_features) arrays."""
    n, seq_len, n_features = X.shape
    flat = scaler.transform(X.reshape(-1, n_features))
    return flat.reshape(n, seq_len, n_features).astype(np.float32)


def compute_class_weights(y_train: NDArray[np.int_]) -> dict[int, float]:
    """Balanced class weights from the training targets (no resampling)."""
    from sklearn.utils.class_weight import compute_class_weight

    weights = compute_class_weight(
        "balanced", classes=np.asarray([0, 1]), y=y_train
    )
    return {0: float(weights[0]), 1: float(weights[1])}


def session_frame_counts(dataset: Dataset) -> dict[str, int]:
    """Number of frames per ``session_id`` (validated rows only)."""
    counts: dict[str, int] = {}
    for index in range(dataset.size):
        session = str(dataset.sessions[index])
        counts[session] = counts.get(session, 0) + 1
    return counts


def sessions_by_class(dataset: Dataset) -> dict[str, list[str]]:
    """Unique ``session_id`` values per label, in first-seen order."""
    result: dict[str, list[str]] = {}
    for index in range(dataset.size):
        label = str(dataset.labels[index])
        session = str(dataset.sessions[index])
        if label not in result:
            result[label] = []
        if session not in result[label]:
            result[label].append(session)
    return result


def recording_dates(session_ids: Iterable[str]) -> set[str]:
    """Distinct recording dates (``YYYYMMDD``) extractable from session ids.

    Collection sessions follow the ``YYYYMMDD_HHMMSS_<hash>`` convention, so a
    single captured day is flagged -- results from one session-day cannot be a
    population estimate. Ids that do not embed a date are simply ignored.
    """
    import re

    dates: set[str] = set()
    for session_id in session_ids:
        match = re.match(r"^(\d{4})-?(\d{2})-?(\d{2})", str(session_id))
        if match:
            dates.add("".join(match.groups()))
    return dates


def preliminary_assessment(
    dataset: Dataset,
    *,
    min_sessions_per_class: int = 5,
) -> tuple[bool, list[str]]:
    """Flag a dataset that cannot support a credible generalization claim.

    Returns ``(usable, reasons)``. ``usable`` is False when any class has fewer
    than ``min_sessions_per_class`` sessions or every session was recorded on a
    single date -- both signal potential overfitting to one participant's
    habits rather than real-world robustness. The reasons string-list explains
    exactly which check failed so results can be labelled PRELIMINARY.

    This is a *reporting* gate, not a training gate: a training gate must also
    verify per-fold sequence counts (:func:`assess_split`).
    """
    reasons: list[str] = []
    per_class = sessions_by_class(dataset)
    for label in LABELS:
        n = len(per_class.get(label, []))
        if n < min_sessions_per_class:
            reasons.append(
                f"{label}: only {n} session(s) (< {min_sessions_per_class}); "
                "results likely reflect one participant's habits, not the population"
            )
    all_ids = {str(dataset.sessions[i]) for i in range(dataset.size)}
    dates = recording_dates(all_ids)
    if len(dates) <= 1:
        label = sorted(dates)[0] if dates else "unknown"
        reasons.append(
            f"data covers a single recording date ({label}): no cross-day or "
            "cross-lighting variance in the estimate"
        )
    return (not reasons), reasons


def assess_split(
    dataset: Dataset,
    folds: dict[str, list[str]],
    *,
    sequence_length: int = 18,
    min_train_per_class: int = 100,
    min_val_per_class: int = 25,
    min_test_per_class: int = 25,
) -> dict:
    """Verify a session-based fold split yields *meaningful* per-class sets.

    Every reported metric should be trusted only if each fold keeps BOTH
    classes AND holds enough sequences of each class for the number to mean
    something. A test fold with ``alert=1`` (the pathology that produced the old
    ``accuracy≈0.83 / roc_auc=1.0 / alert support=1`` report) is worse than no
    evaluation: it looks real but generalizes to nothing.

    The fold session ids are already disjoint (the caller builds them), so this
    only computes per-class sequence counts -- ``max(0, frames - len + 1)`` per
    session, identical to :func:`build_sequences` with ``step=1`` -- and checks
    them against the requested minima (``train`` / ``validation`` / ``test``).

    Returns a dict with:

    * ``adequate`` -- all folds/classes above their minimum;
    * ``reasons`` -- one human-readable string per violation;
    * ``minima`` / ``sequence_length`` -- the thresholds applied;
    * ``folds`` -- ``{fold: {session_ids, per_class: {label: {session_ids,
      n_sessions, n_sequences}}}}`` so the caller can print the full class
      balance BEFORE training and persist it.

    ``adequate=False`` means the caller should STOP and report rather than fit
    an LSTM whose train/val/test sets are too small or single-class.
    """
    minima = {
        "train": min_train_per_class,
        "validation": min_val_per_class,
        "test": min_test_per_class,
    }
    per_class_sessions = sessions_by_class(dataset)
    frame_counts = session_frame_counts(dataset)
    report: dict = {
        "adequate": True,
        "reasons": [],
        "minima": minima,
        "sequence_length": sequence_length,
        "folds": {},
    }
    for fold, ids in (
        ("train", list(folds["train"])),
        ("validation", list(folds["validation"])),
        ("test", list(folds["test"])),
    ):
        sequences_by_label = {label: 0 for label in LABELS}
        skipped: list[str] = []
        for session in ids:
            label = next(
                (
                    label
                    for label in LABELS
                    if session in per_class_sessions.get(label, [])
                ),
                None,
            )
            if label is None:
                continue
            frames = frame_counts.get(session, 0)
            if frames < sequence_length:
                skipped.append(f"{session} ({frames} frames)")
                continue
            sequences_by_label[label] += frames - sequence_length + 1
        fold_report: dict = {"session_ids": sorted(ids), "per_class": {}}
        for label in LABELS:
            fold_sessions = sorted(
                s for s in ids if s in per_class_sessions.get(label, [])
            )
            n_seq = sequences_by_label[label]
            fold_report["per_class"][label] = {
                "session_ids": fold_sessions,
                "n_sessions": len(fold_sessions),
                "n_sequences": n_seq,
            }
            if n_seq < minima[fold]:
                report["adequate"] = False
                report["reasons"].append(
                    f"{fold.upper()} fold: {label} -> {n_seq} sequence(s) "
                    f"(minimum required {minima[fold]}) from sessions "
                    f"{fold_sessions or 'NONE'}"
                )
        if skipped:
            report["reasons"].append(
                f"{fold.upper()} fold: {len(skipped)} session(s) shorter than "
                f"sequence_length={sequence_length} contributed no sequences: "
                f"{skipped}"
            )
        report["folds"][fold] = fold_report
    return report


def format_sequences_report(
    dataset: Dataset, train_ids: list[str], val_ids: list[str], test_ids: list[str],
    sequence_length: int,
) -> str:
    """Human-readable summary of the session split + per-fold sequence counts."""
    lines: list[str] = []
    n_all = len({str(s) for s in dataset.sessions})
    for fold, ids in (
        ("TRAIN", train_ids),
        ("VALIDATION", val_ids),
        ("TEST", test_ids),
    ):
        _, _, stats = build_sequences(
            dataset, ids, sequence_length=sequence_length
        )
        lines.append(
            f"{fold} sessions ({len(ids)} of {n_all}): {sorted(ids)} -> "
            f"{stats['n_sequences']} sequences "
            f"({stats['n_sequences_alert']} alert / {stats['n_sequences_drowsy']} drowsy)"
        )
    return "\n".join(lines)


__all__ = [
    "LABEL_MODES",
    "group_by_session",
    "build_sequences",
    "split_sessions",
    "fit_scaler",
    "scale_sequences",
    "compute_class_weights",
    "session_frame_counts",
    "sessions_by_class",
    "recording_dates",
    "preliminary_assessment",
    "assess_split",
    "format_sequences_report",
]

