"""Tests for dataset I/O + validation (src.ml.dataset)."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.ml.dataset import (
    DatasetError,
    describe_missing,
    drop_invalid_rows,
    group_train_test_split,
    load_dataset,
    validate_dataset,
)
from src.ml.schema import HEADER


def _write_rows(path, rows, columns=None):
    columns = columns or list(HEADER)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _good_row(session="s1", label="alert", idx=0, pitch="2.0"):
    return {
        "timestamp": str(idx),
        "session_id": session,
        "ear_left": "0.29",
        "ear_right": "0.31",
        "ear_mean": "0.30",
        "mar": "0.10",
        "perclos": "12.0",
        "eye_closed_duration": "0.0",
        "yawn_duration": "0.0",
        "pitch": pitch,
        "yaw": "-3.0",
        "roll": "1.0",
        "label": label,
    }


class TestLoadDataset:
    def test_loads_valid_rows(self, tmp_path):
        path = tmp_path / "data.csv"
        _write_rows(
            path,
            [_good_row("s1", "alert", 0), _good_row("s1", "drowsy", 1)],
        )
        data = load_dataset(path)
        assert data.size == 2
        assert data.features.shape == (2, 10)
        assert list(data.labels) == ["alert", "drowsy"]
        assert list(data.sessions) == ["s1", "s1"]

    def test_normalizes_label_case(self, tmp_path):
        path = tmp_path / "data.csv"
        row = _good_row(idx=0)
        row["label"] = "DROWSY"
        _write_rows(path, [row])
        data = load_dataset(path)
        assert data.labels[0] == "drowsy"

    def test_skips_invalid_labels(self, tmp_path):
        path = tmp_path / "data.csv"
        rows = [_good_row(idx=0)]
        rows.append({**_good_row(idx=1), "label": "napping"})
        _write_rows(path, rows)
        data = load_dataset(path)
        assert data.size == 1

    def test_empty_string_feature_becomes_nan(self, tmp_path):
        path = tmp_path / "data.csv"
        row = _good_row(idx=0)
        row["pitch"] = ""
        _write_rows(path, [row])
        data = load_dataset(path)
        assert np.isnan(data.features[0, 7])

    def test_missing_required_column_raises(self, tmp_path):
        path = tmp_path / "data.csv"
        _write_rows(path, [_good_row()], columns=("timestamp", "label"))
        with pytest.raises(DatasetError, match="missing required column"):
            load_dataset(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(tmp_path / "nope.csv")

    def test_merges_multiple_files(self, tmp_path):
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        _write_rows(a, [_good_row("s1", idx=0)])
        _write_rows(b, [_good_row("s2", idx=0)])
        data = load_dataset([a, b])
        assert data.size == 2
        assert set(data.sessions) == {"s1", "s2"}


class TestDropInvalidRows:
    def test_drops_nan_and_empty_session(self, tmp_path):
        path = tmp_path / "data.csv"
        rows = [_good_row("s1", idx=0)]
        bad = _good_row("s1", idx=1)
        bad["mar"] = ""
        rows.append(bad)
        no_session = _good_row("", idx=2)
        rows.append(no_session)
        _write_rows(path, rows)
        data, dropped = drop_invalid_rows(load_dataset(path))
        assert dropped == 2
        assert data.size == 1

    def test_keeps_finite_rows(self, tmp_path):
        path = tmp_path / "data.csv"
        _write_rows(path, [_good_row(idx=0), _good_row(idx=1)])
        data, dropped = drop_invalid_rows(load_dataset(path))
        assert dropped == 0
        assert data.size == 2


class TestDescribeMissing:
    def test_counts_nan_per_column(self, tmp_path):
        path = tmp_path / "data.csv"
        rows = [_good_row(idx=0)]
        bad_pitch = _good_row(idx=1)
        bad_pitch["pitch"] = ""
        rows.append(bad_pitch)
        bad_mar = _good_row(idx=2)
        bad_mar["mar"] = "not-a-number"
        rows.append(bad_mar)
        _write_rows(path, rows)
        data = load_dataset(path)
        missing = describe_missing(data)
        assert missing["columns"]["pitch"] == 1
        assert missing["columns"]["mar"] == 1
        assert missing["columns"]["ear_left"] == 0
        assert missing["empty_session"] == 0

    def test_counts_inf_as_invalid(self, tmp_path):
        path = tmp_path / "data.csv"
        row = _good_row(idx=0)
        row["roll"] = "inf"
        _write_rows(path, [row])
        data = load_dataset(path)
        missing = describe_missing(data)
        assert missing["columns"]["roll"] == 1

    def test_counts_empty_session(self, tmp_path):
        path = tmp_path / "data.csv"
        _write_rows(path, [_good_row("", idx=0), _good_row("", idx=1)])
        data = load_dataset(path)
        missing = describe_missing(data)
        assert missing["empty_session"] == 2


class TestValidateDataset:
    def test_empty_is_flagged(self):
        from src.ml.dataset import Dataset

        data = Dataset(
            features=np.empty((0, 10)),
            labels=np.empty((0,), dtype=str),
            sessions=np.empty((0,), dtype=str),
            timestamps=np.empty((0,)),
            sources=[],
        )
        issues = validate_dataset(data)
        assert any("empty" in issue for issue in issues)

    def test_small_and_single_session_warnings(self, tmp_path):
        path = tmp_path / "data.csv"
        _write_rows(path, [_good_row("s1", idx=0), _good_row("s1", idx=1)])
        issues = validate_dataset(load_dataset(path))
        assert any("very small" in issue for issue in issues)
        assert any("only one session" in issue for issue in issues)


class TestGroupSplitNoLeakage:
    def test_test_sessions_never_in_train(self, tmp_path):
        pytest.importorskip("sklearn")
        path = tmp_path / "data.csv"
        rows = []
        idx = 0
        for s in range(1, 9):
            for _ in range(6):
                rows.append(_good_row(f"session-{s}", "alert" if s % 2 else "drowsy", idx))
                idx += 1
        _write_rows(path, rows)
        data = load_dataset(path)
        train_idx, test_idx = group_train_test_split(data, random_state=7)
        train_sessions = set(data.sessions[train_idx])
        test_sessions = set(data.sessions[test_idx])
        assert train_sessions
        assert test_sessions
        assert not (train_sessions & test_sessions)

    def test_similar_split_sizes(self, tmp_path):
        pytest.importorskip("sklearn")
        path = tmp_path / "data.csv"
        rows = [_good_row(f"session-{s}", idx=i) for s in range(1, 6) for i in range(4)]
        _write_rows(path, rows)
        data = load_dataset(path)
        train_idx, test_idx = group_train_test_split(
            data, test_size=0.25, random_state=0
        )
        ratio = len(test_idx) / max(len(train_idx) + len(test_idx), 1)
        assert 0.0 < ratio <= 0.75
        assert len(train_idx) + len(test_idx) == data.size
