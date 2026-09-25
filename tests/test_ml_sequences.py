"""Temporal sequence building: grouping, windows, fold splitting, scaling."""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.dataset import Dataset, DatasetError
from src.ml.schema import N_FEATURES
from src.ml.sequences import (
    assess_split,
    build_sequences,
    compute_class_weights,
    fit_scaler,
    preliminary_assessment,
    recording_dates,
    scale_sequences,
    split_sessions,
)


def make_dataset(
    session_sizes: list[tuple[str, str, int]],
    feature: float = 0.5,
) -> Dataset:
    """Build a Dataset with one session per (id, label, size) tuple.

    Feature vectors are constant ``feature`` and timestamps are 1, 2, 3... per
    session so chronological ordering can be asserted.
    """
    bases: list[tuple[int, float]] = []
    for _session_id, _label, size in session_sizes:
        for k in range(size):
            bases.append((k + 1, feature))
    total = sum(s for *_s, s in session_sizes)
    features = np.asarray([[feature] * N_FEATURES] * total, dtype=np.float32)
    sessions = np.empty(total, dtype=object)
    labels = np.empty(total, dtype=object)
    timestamps = np.empty(total, dtype=np.float64)
    start = 0
    for session_id, label, size in session_sizes:
        for k in range(size):
            sessions[start + k] = session_id
            labels[start + k] = label
            timestamps[start + k] = k + 1
        start += size
    return Dataset(
        features=features,
        labels=labels,
        sessions=sessions,
        timestamps=timestamps,
        sources=["synthetic.csv"],
    )


def sorted_timestamps_for(dataset: Dataset, session_id: str) -> np.ndarray:
    idx = [i for i in range(dataset.size) if str(dataset.sessions[i]) == session_id]
    return dataset.timestamps[idx]


class TestSplitSessions:
    def test_balanced_folds_cover_both_classes(self):
        dataset = make_dataset(
            [
                ("a1", "alert", 50),
                ("a2", "alert", 50),
                ("a3", "alert", 50),
                ("d1", "drowsy", 50),
                ("d2", "drowsy", 50),
                ("d3", "drowsy", 50),
            ]
        )
        folds = split_sessions(
            dataset, test_per_class=1, val_per_class=1, random_state=1
        )
        all_sessions = ["a1", "a2", "a3", "d1", "d2", "d3"]
        assert len(folds["test"]) == len(folds["validation"]) == 2
        assert len(folds["train"]) == 2
        union = folds["test"] + folds["validation"] + folds["train"]
        assert sorted(union) == all_sessions, "folds must partition all sessions"
        assert len(set(union)) == len(union), "folds must be disjoint"
        for fold in folds.values():
            seen = {
                str(dataset.labels[i])
                for i in range(dataset.size)
                if str(dataset.sessions[i]) in fold
            }
            assert seen == {"alert", "drowsy"}, f"{fold} lost a class"

    def test_reproducible_with_fixed_seed(self):
        dataset = make_dataset(
            [
                ("a1", "alert", 50),
                ("a2", "alert", 50),
                ("a3", "alert", 50),
                ("d1", "drowsy", 50),
                ("d2", "drowsy", 50),
                ("d3", "drowsy", 50),
            ]
        )
        first = split_sessions(dataset, random_state=7)
        again = split_sessions(dataset, random_state=7)
        assert first == again

    def test_too_few_sessions_raises(self):
        dataset = make_dataset([("a1", "alert", 50), ("d1", "drowsy", 50)])
        with pytest.raises(DatasetError):
            split_sessions(dataset, test_per_class=1, val_per_class=0)


class TestGroupAndBuild:
    def test_windows_never_cross_sessions(self):
        dataset = make_dataset(
            [("a1", "alert", 10), ("a2", "drowsy", 12), ("a3", "alert", 14)]
        )
        X, y, stats = build_sequences(dataset, ["a1", "a2", "a3"], sequence_length=4)
        assert X.shape == (10 - 4 + 1 + 12 - 4 + 1 + 14 - 4 + 1, 4, N_FEATURES)
        assert stats["sessions_skipped_short"] == 0
        assert set(np.unique(y).tolist()) == {0, 1}

    def test_short_session_skipped_and_reported(self):
        dataset = make_dataset(
            [("a1", "alert", 6), ("d1", "drowsy", 40)]
        )
        X, y, stats = build_sequences(dataset, ["a1", "d1"], sequence_length=20)
        assert stats["sessions_skipped_short"] == 1
        assert stats["n_sequences_alert"] == 0
        assert stats["n_sequences_drowsy"] == 40 - 20 + 1

    def test_too_short_everywhere_raises(self):
        dataset = make_dataset([("a1", "alert", 5), ("d1", "drowsy", 5)])
        with pytest.raises(DatasetError):
            build_sequences(dataset, ["a1", "d1"], sequence_length=10)

    def test_label_is_alert_zero_drowsy_one(self):
        dataset = make_dataset([("a1", "alert", 8), ("d1", "drowsy", 8)])
        X_al, y_al, _ = build_sequences(dataset, ["a1"], sequence_length=3)
        X_dr, y_dr, _ = build_sequences(dataset, ["d1"], sequence_length=3)
        assert (y_al == 0).all()
        assert (y_dr == 1).all()

    def test_chronological_ordering_asserted(self):
        dataset = make_dataset([("d1", "drowsy", 20)])
        first_idx = sorted_timestamps_for(dataset, "d1")
        assert (np.diff(first_idx) > 0).all()
        assert np.allclose(first_idx, np.arange(1, 21))

    def test_step_produces_fewer_windows(self):
        dataset = make_dataset([("a1", "alert", 20)])
        _, _, stats_step1 = build_sequences(dataset, ["a1"], sequence_length=4, step=1)
        _, _, stats_step2 = build_sequences(dataset, ["a1"], sequence_length=4, step=2)
        assert stats_step1["n_sequences"] == 17
        assert stats_step2["n_sequences"] == 9


class TestSplitAdequacy:
    def _balanced_dataset(self):
        # 6 sessions per class, each 60 frames -> 43 sequences per session.
        sessions = [
            (f"a{i}", "alert", 60) for i in range(6)
        ] + [(f"d{i}", "drowsy", 60) for i in range(6)]
        return make_dataset(sessions)

    def test_adequate_split_passes_default_minima(self):
        dataset = self._balanced_dataset()
        folds = split_sessions(
            dataset, test_per_class=1, val_per_class=2, random_state=1
        )
        report = assess_split(dataset, folds, sequence_length=18)
        assert report["adequate"] is True
        assert report["reasons"] == []
        for fold in ("test", "validation", "train"):
            for label in ("alert", "drowsy"):
                entry = report["folds"][fold]["per_class"][label]
                assert entry["n_sessions"] > 0
                assert entry["n_sequences"] >= report["minima"][fold]

    def test_tiny_test_class_is_flag_not_silent(self):
        # Mirrors the real pathology: the only large alert session teaches the
        # model, and the held-out alert session supports a single sequence.
        dataset = make_dataset(
            [
                ("a1", "alert", 300),
                ("a2", "alert", 20),
                ("a3", "alert", 20),
                ("a4", "alert", 18),
                ("d1", "drowsy", 300),
                ("d2", "drowsy", 300),
                ("d3", "drowsy", 40),
                ("d4", "drowsy", 40),
            ]
        )
        folds = split_sessions(
            dataset, test_per_class=1, val_per_class=1, random_state=42
        )
        report = assess_split(dataset, folds, sequence_length=18)
        assert report["adequate"] is False
        assert any(
            "TEST fold" in r and "alert" in r and "sequence(s)" in r
            for r in report["reasons"]
        )
        test_alert = report["folds"]["test"]["per_class"]["alert"]
        assert test_alert["n_sequences"] < report["minima"]["test"]

    def test_missing_class_in_test_reported(self):
        dataset = make_dataset([("a1", "alert", 60), ("d1", "drowsy", 60)])
        folds = {"train": ["a1", "d1"], "validation": ["a1"], "test": ["d1"]}
        report = assess_split(dataset, folds, sequence_length=18)
        assert report["adequate"] is False
        assert any(
            "TEST fold" in r and "alert" in r and "NONE" in r
            for r in report["reasons"]
        )

    def test_short_sessions_reported_as_skipped(self):
        dataset = make_dataset(
            [
                ("a1", "alert", 60),
                ("a2", "alert", 60),
                ("a3", "alert", 60),
                ("a4", "alert", 5),
                ("d1", "drowsy", 60),
                ("d2", "drowsy", 60),
                ("d3", "drowsy", 60),
                ("d4", "drowsy", 5),
            ]
        )
        folds = {
            "train": ["a1", "a2", "a3", "d1", "d2", "d3"],
            "validation": ["a4", "d1"],
            "test": ["a4", "d4"],
        }
        report = assess_split(dataset, folds, sequence_length=18)
        assert any("shorter than sequence_length=18" in r for r in report["reasons"])


class TestPreliminaryAndDates:
    def test_recording_dates_parsed_from_ids(self):
        assert recording_dates(
            ["20260914_224436_7bfbe7", "2026-09-15_001122_x", "demo-01"]
        ) == {"20260914", "20260915"}

    def test_single_class_small_is_preliminary(self):
        dataset = make_dataset([("a1", "alert", 60), ("d1", "drowsy", 60)])
        usable, reasons = preliminary_assessment(dataset)
        assert usable is False
        joined = " | ".join(reasons).lower()
        assert "drowsy" in joined and "1 session" in joined

    def test_many_sessions_same_date_is_preliminary(self):
        dataset = make_dataset(
            [
                (f"20260914_{i}0000_abc", "alert", 60) for i in range(8)
            ]
            + [
                (f"20260914_{i}0000_def", "drowsy", 60) for i in range(8)
            ]
        )
        usable, reasons = preliminary_assessment(dataset)
        assert usable is False
        assert any("single recording date" in r for r in reasons)

    def test_multi_date_multi_class_not_preliminary(self):
        dataset = make_dataset(
            [
                (f"2026091{i}_100000_a", "alert", 60) for i in range(1, 7)
            ]
            + [
                (f"2026091{i}_100000_b", "drowsy", 60) for i in range(1, 7)
            ]
        )
        usable, reasons = preliminary_assessment(dataset)
        assert usable is True
        assert reasons == []


class TestScalerAndWeights:
    def test_scaler_fit_on_train_only_statistics(self):
        train = make_dataset([("a1", "alert", 30)], feature=0.2)
        test = make_dataset([("d1", "drowsy", 30)], feature=0.8)
        X_train, _, _ = build_sequences(train, ["a1"], sequence_length=10)
        X_test, _, _ = build_sequences(test, ["d1"], sequence_length=10)
        scaler = fit_scaler(X_train)
        scaled_train = scale_sequences(X_train, scaler)
        scaled_test = scale_sequences(X_test, scaler)
        assert scaled_train.shape == X_train.shape
        assert scaled_test.shape[1:] == (10, N_FEATURES)

    def test_class_weights_balanced_from_train(self):
        y_train = np.array([0] * 8 + [1] * 2)
        weights = compute_class_weights(y_train)
        assert weights[0] < weights[1]
        assert np.isclose(weights[0] * 8, weights[1] * 2)
