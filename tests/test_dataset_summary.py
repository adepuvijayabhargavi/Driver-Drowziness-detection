"""Tests for the dataset summary tool (scripts.dataset_summary)."""

from __future__ import annotations

import csv
import re

from scripts.dataset_summary import print_dataset_summary
from src.ml.schema import HEADER


def _write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HEADER))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(session="s1", label="alert", idx=0, pitch="2.0"):
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


def test_summary_reports_missing_dataset(tmp_path, capsys):
    missing = tmp_path / "nope.csv"
    print_dataset_summary([str(missing)])
    out = capsys.readouterr().out
    assert "MISSING" in out
    assert "No usable dataset found" in out


def test_summary_reports_counts_balance_and_sessions(tmp_path, capsys):
    path = tmp_path / "drowsiness_dataset.csv"
    rows = []
    idx = 0
    for session, label in (
        ("A1", "alert"), ("D1", "drowsy"),
        ("A2", "alert"), ("D2", "drowsy"),
    ):
        for _ in range(10):
            rows.append(_row(session, label, idx))
            idx += 1
    bad = _row("A1", "alert", idx)
    bad["pitch"] = ""
    rows.append(bad)
    _write_rows(path, rows)

    print_dataset_summary([str(path)])
    out = capsys.readouterr().out

    assert re.search(r"usable for training\s+: 40", out)
    assert re.search(r"alert\s+20", out)
    assert re.search(r"drowsy\s+20", out)
    assert "4 unique" in out
    assert re.search(r"pitch\s+1", out)
