"""Print a clear, read-only summary of the ML drowsiness dataset.

Before training you inspect the collected data: how many ALERT / DROWSY rows,
how many sessions, how many missing/invalid values per feature column, whether
both classes appear in several sessions, and whether a session-based split is
feasible. This script only reads -- it never modifies or deletes data.

Usage::

    python scripts/dataset_summary.py
    python scripts/dataset_summary.py data/features/drowsiness_dataset.csv
    python scripts/dataset_summary.py data/features/*.csv
"""

from __future__ import annotations

import argparse
import csv
import glob as _glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import FEATURES_DIR
from src.ml.dataset import (
    DatasetError,
    describe_missing,
    drop_invalid_rows,
    load_dataset,
    validate_dataset,
)
from src.ml.schema import FEATURE_COLUMNS, LABELS

DEFAULT_PATHS = (str(FEATURES_DIR / "drowsiness_dataset.csv"),)


def _total_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _is_synthetic_demo(path: str) -> bool:
    name = Path(path).name.lower()
    return "demo" in name or "synthetic" in name


def _warn_line(path: str) -> str:
    if _is_synthetic_demo(path):
        return "  [synthetic demo - NOT real driver data]"
    return ""


def _print_split_feasibility(
    sessions_per_class: dict[str, set[str]],
    n_sessions: int,
) -> None:
    print("\nsplit feasibility (session-based, no leakage):")
    if n_sessions < 2:
        print("  NOT READY - fewer than 2 sessions; a session split is impossible.")
    else:
        per_class_bad = {
            lab: n for lab, n in sessions_per_class.items() if len(n) < 2
        }
        if per_class_bad:
            for lab, count in per_class_bad.items():
                print(
                    f"  CAUTION - {lab} appears in {count} session(s) only; "
                    "a held-out session of that class would leave a single-class "
                    "fold (metrics unstable)."
                )
        if n_sessions >= 4 and not per_class_bad:
            print("  OK - several sessions per class; hold out whole sessions.")


def print_dataset_summary(paths: list[str]) -> None:
    print("Dataset summary (real-time ML drowsiness data)")
    print("=" * 52)
    present: list[str] = []
    print("source files:")
    for spec in paths:
        expanded = _glob.glob(spec)
        if not expanded and Path(spec).is_file():
            expanded = [spec]
        if not expanded:
            print(f"  {spec:<52} MISSING")
            continue
        for raw in expanded:
            path = Path(raw)
            if not path.is_file():
                continue
            rows = _total_rows(path)
            size = path.stat().st_size
            print(
                f"  {str(path):<42} {rows:>7,} rows "
                f"({size / 1024:,.1f} KiB){_warn_line(str(path))}"
            )
            present.append(str(path))
    print()

    if not present:
        print("No usable dataset found. Collect REAL data first, e.g.:")
        print("  python scripts/collect_ml_data.py --label alert  --samples 300")
        print("  python scripts/collect_ml_data.py --label drowsy --samples 300")
        print("Then re-run this summary before training.")
        return

    try:
        raw = load_dataset(present)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return

    cleaned, dropped = drop_invalid_rows(raw)
    issues = validate_dataset(cleaned)

    print("rows:")
    print(f"  loaded (valid labels) : {raw.size:,}")
    print(f"  dropped (invalid)     : {dropped:,}")
    print(f"  usable for training   : {cleaned.size:,}")

    if dropped:
        print("\nmissing/invalid values per feature column:")
        missing = describe_missing(raw)
        for column in FEATURE_COLUMNS:
            count = missing["columns"][column]
            if count:
                pct = 100.0 * count / max(raw.size, 1)
                print(f"  {column:<22} {count:>8,}  ({pct:.2f}%)")
        if missing["empty_session"]:
            n = missing["empty_session"]
            pct = 100.0 * n / max(raw.size, 1)
            print(f"  {'session_id (empty)':<22} {n:>8,}  ({pct:.2f}%)")

    print("\nclass balance:")
    counts = {
        lab: int((cleaned.labels == lab).sum()) for lab in LABELS
    }
    for lab in LABELS:
        pct = 100.0 * counts[lab] / max(cleaned.size, 1)
        print(f"  {lab:<10} {counts[lab]:>7,}  ({pct:.1f}%)")
    if cleaned.size and min(counts.values()) / max(counts.values()) < 0.33:
        print("  NOTE: strong class imbalance - consider collecting more of the minority class.")

    unique_sessions = sorted(set(cleaned.sessions))
    n_sessions = len(unique_sessions)
    print(f"\nsessions: {n_sessions} unique")
    sessions_per_class: dict[str, set[str]] = {
        lab: set() for lab in LABELS
    }
    per_session: dict[str, Counter] = defaultdict(Counter)
    for lab, session in zip(cleaned.labels, cleaned.sessions, strict=True):
        sessions_per_class[lab].add(session)
        per_session[session][lab] += 1

    width = max((len(s) for s in unique_sessions), default=1)
    header = f"  {'session_id':<{width}}   "
    for lab in LABELS:
        header += f"{lab:>10}"
    print(header)
    for session in unique_sessions:
        line = f"  {session:<{width}}   "
        for lab in LABELS:
            line += f"{per_session[session][lab]:>10}"
        print(line)

    for lab in LABELS:
        print(
            f"  {lab}: present in {len(sessions_per_class[lab])} "
            f"session(s)"
        )

    _print_split_feasibility(sessions_per_class, n_sessions)

    if cleaned.size < 1000 or n_sessions < 4:
        print("\nNOTE: dataset is still small.")
        print("      \"Preliminary results - larger multi-session/multi-participant")
        print("      data is required for reliable generalization.\"")
    if issues:
        print("\ndataset checks:")
        for issue in issues:
            print(f"  - {issue}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_PATHS),
        help="Dataset CSV(s) or globs. Default: the collector output file.",
    )
    args = parser.parse_args(argv)
    print_dataset_summary(args.paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
