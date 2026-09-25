"""Train the optional ML drowsiness classifier from recorded sessions.

Reads one or more CSVs produced by ``scripts/record_session.py`` (or any CSV
with the columns ``ear``, ``mar`` and ``state``), featurizes each row with a
rolling window, and fits a Random Forest on two classes: ALERT vs DROWSY.
Rows labelled YAWNING/UNKNOWN/NO_FACE are excluded from training.

The trained model is saved as joblib and then loaded automatically at runtime
when ``python app.py --ml`` is used.

Usage::

    python scripts/train_classifier.py \\
        --data data/sessions/*.csv \\
        --out data/models/drowsiness_rf.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.ml.features import extract_batch_features


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse one session CSV into (ear, mar, state) arrays."""
    import csv

    ears, mars, states = [], [], []
    with path.open(newline="") as fh:
        rows = csv.DictReader(fh, fieldnames=["timestamp", "ear", "mar", "state"])
        next(rows, None)  # skip header
        for r in rows:
            if not r["ear"] or not r["mar"]:
                continue
            try:
                ears.append(float(r["ear"]))
                mars.append(float(r["mar"]))
                states.append(str(r["state"]).strip().upper())
            except ValueError:
                continue
    return (
        np.asarray(ears, dtype=np.float64),
        np.asarray(mars, dtype=np.float64),
        np.asarray(states),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True, help="Session CSV(s).")
    parser.add_argument("--out", default="data/models/drowsiness_rf.joblib")
    parser.add_argument("--window", type=int, default=45)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Missing ML deps. Run: pip install -r requirements-ml.txt")
        return 1

    X_parts: list[np.ndarray] = []
    y_parts: list[str] = []
    for pattern in args.data:
        is_glob = any(c in pattern for c in "*?[")
        paths = sorted(Path.cwd().glob(pattern)) if is_glob else [Path(pattern)]
        for path in paths:
            if not path.exists():
                print(f"Skipping missing file: {path}")
                continue
            ear, mar, states = load_csv(path)
            feats = extract_batch_features(ear, mar, window=args.window)
            # Keep rows with a warm window and a usable label.
            mask = ~np.isnan(feats[:, 0])
            valid = np.where(mask & np.isin(states, ["ALERT", "DROWSY"]))[0]
            if valid.size == 0:
                print(f"No usable rows in {path} (need ALERT/DROWSY labels).")
                continue
            X_parts.append(feats[valid])
            y_parts.extend(states[valid].tolist())
            print(f"{path}: {valid.size} labelled rows")

    if not X_parts:
        print("No training data. Record sessions first via scripts/record_session.py")
        return 1

    X = np.vstack(X_parts)
    y = np.asarray(y_parts)
    print(f"Training on {X.shape[0]} samples, {X.shape[1]} features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=args.seed, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("\n=== Metrics ===")
    print(f"accuracy : {accuracy_score(y_test, pred):.3f}")
    print(f"F1 (macro): {f1_score(y_test, pred, average='macro'):.3f}")
    import sklearn
    labels = sorted(set(y_test))
    print("confusion matrix:")
    print(sklearn.metrics.classification_report(y_test, pred, labels=labels, zero_division=0))
    print("feature importances:", dict(zip(
        ["ear_mean","ear_min","ear_q10","ear_std","mar_mean","mar_max","mar_std",
         "eye_closed_fraction","mouth_open_fraction","mar_range"],
        model.feature_importances_.round(3),
    )))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"\nModel saved to {out}")
    print("Enable at runtime:  python app.py --ml --ml-model " + str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
