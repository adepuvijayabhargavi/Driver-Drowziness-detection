"""Train and evaluate the temporal deep-learning (LSTM) drowsiness classifier.

The DL model learns ALERT vs DROWSY from *sequences* of consecutive frame
feature vectors (default 18 frames -- the window the deployed ``app.py --dl``
model expects) instead of single frames, so it can capture temporal patterns
such as slow eye-closure drift, blinking rhythm collapse and gradual
head-nodding.

Honesty rules (same as ``ml/train.py``):

* only the REAL collected dataset is used -- the ``demo_synthetic_dataset.csv``
  file is skipped by default and is never allowed to influence reported
  metrics (pass ``--include-synthetic`` only for scratch exploration);
* sessions are split into TRAIN / VALIDATION / TEST *before* sequences are
  created, so no overlapping window can leak a session into two folds;
* the split is deterministic (seeded + size-ordered per class), so re-runs on
  the same data reproduce the same folds;
* a feasibility gate checks the per-class sequence counts in EVERY fold BEFORE
  training. if any fold cannot keep both classes with enough sequences, the
  script STOPS (exit code 3) and writes an honest report instead of producing
  an accuracy/ROC-AUC number that means nothing (e.g. the old report with
  ``alert support=1``, ``roc_auc=1.0`` and a 1.0 validation accuracy every
  epoch);
* the StandardScaler is fitted on the training sequences only;
* early stopping tracks the validation loss and restores the best weights;
* every reported number is measured on the completely unseen test sessions;
* small / single-day / single-participant datasets are clearly labelled
  PRELIMINARY.

Usage::

    python scripts/train_deep_learning.py --force
    python scripts/train_deep_learning.py --epochs 30
    python scripts/train_deep_learning.py --sequence-length 30  # non-standard

Exit codes: 0 = trained and evaluated; 3 = STOPPED, the current real data
cannot produce a reliable session-based split (see ``results/deep_learning_metrics.json``).

Keras runs on a CPU-friendly backend (torch) by default -- no GPU required.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.config.settings import FEATURES_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402
from src.ml.dataset import (  # noqa: E402
    DatasetError,
    describe_missing,
    drop_invalid_rows,
    load_dataset,
    validate_dataset,
)
from src.ml.schema import FEATURE_COLUMNS, LABELS  # noqa: E402
from src.ml.sequences import (  # noqa: E402
    assess_split,
    build_sequences,
    compute_class_weights,
    fit_scaler,
    preliminary_assessment,
    recording_dates,
    scale_sequences,
    session_frame_counts,
    sessions_by_class,
    split_sessions,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_OUT = MODELS_DIR / "drowsiness_lstm.keras"
DEFAULT_SCALER_OUT = MODELS_DIR / "drowsiness_lstm_scaler.joblib"
DEFAULT_METADATA_OUT = MODELS_DIR / "drowsiness_lstm_metadata.json"


def build_lstm_model(
    sequence_length: int,
    n_features: int,
    *,
    units: tuple[int, int] = (64, 32),
    dense_units: int = 16,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    random_state: int = 42,
):
    """Compile the LSTM binary classifier requested in the design."""
    import keras
    from keras import layers

    keras.utils.set_random_seed(random_state)
    model = keras.Sequential(
        [
            layers.Input(shape=(sequence_length, n_features)),
            layers.LSTM(units[0], return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(units[1]),
            layers.Dropout(dropout),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    """Accuracy / per-class P/R/F1 / macro / weighted / ROC-AUC."""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    result: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(
            roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else None
        ),
    }
    for label, index in (("alert", 0), ("drowsy", 1)):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[index], zero_division=0
        )
        result[label] = {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        }
    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=[0, 1], zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=[0, 1], zero_division=0
    )
    result["macro_precision"], result["macro_recall"], result["macro_f1"] = (
        float(macro[0]),
        float(macro[1]),
        float(macro[2]),
    )
    result["weighted_precision"], result["weighted_recall"], result["weighted_f1"] = (
        float(weighted[0]),
        float(weighted[1]),
        float(weighted[2]),
    )
    return result


def _maybe_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise DatasetError(
            f"{path} already exists. Re-run with --force to overwrite."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def _retrain_random_forest(X_train_frames, y_train_frames, X_test_frames, y_test_frames):
    """Re-train the reference Random Forest on the SAME session split so the
    ML-vs-DL comparison is apples-to-apples (same train/test sessions)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )
    estimator.fit(X_train_frames, y_train_frames)
    proba = estimator.predict_proba(X_test_frames)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    return classification_metrics(y_test_frames, y_pred, proba)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        nargs="*",
        default=[str(FEATURES_DIR / "drowsiness_dataset.csv")],
        help="Dataset CSV(s). Glob patterns are expanded.",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=18,
        help="Consecutive frames per temporal window. The deployed "
        "app.py --dl model and its metadata use 18; changing it forces a "
        "model that must be deployed together with the matching window.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-sessions-per-class", type=int, default=1)
    parser.add_argument("--val-sessions-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Also load demo/synthetic CSVs. NEVER used for reported metrics.",
    )
    parser.add_argument(
        "--min-train-per-class", type=int, default=100,
        help="Minimum sequences of EACH class required in TRAIN before an LSTM "
        "can learn (below this we refuse to report metrics).",
    )
    parser.add_argument(
        "--min-val-per-class", type=int, default=25,
        help="Minimum sequences of EACH class required in VALIDATION so early "
        "stopping is not driven by a handful of windows.",
    )
    parser.add_argument(
        "--min-test-per-class", type=int, default=25,
        help="Minimum sequences of EACH class required in the final TEST set "
        "so per-class precision/recall/F1 and the confusion matrix are "
        "meaningful.",
    )
    parser.add_argument("--units", type=str, default="64,32",
                        help="Comma-separated LSTM layer sizes.")
    parser.add_argument("--dense-units", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--out", default=str(DEFAULT_MODEL_OUT))
    parser.add_argument("--scaler-out", default=str(DEFAULT_SCALER_OUT))
    parser.add_argument("--metadata-out", default=str(DEFAULT_METADATA_OUT))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing model/scaler/metadata.")
    args = parser.parse_args(argv)

    # ----------------------------------------------------------- load data --
    expanded: list[str] = []
    for spec in args.data:
        if Path(spec).is_absolute():
            expanded.append(spec)
            continue
        matches = sorted(Path(".").glob(spec))
        if matches:
            expanded.extend(str(p) for p in matches)
        elif Path(spec).exists():
            expanded.append(spec)
        else:
            expanded.append(spec)
    from collections import OrderedDict

    data_specs = list(OrderedDict.fromkeys(expanded))

    if not args.include_synthetic:
        real_only = [
            spec
            for spec in data_specs
            if "synthetic" not in Path(spec).name.lower()
        ]
        for spec in data_specs:
            if spec not in real_only:
                print(
                    f"[dataset] SKIPPED synthetic file: {spec} "
                    "(demo data must never feed reported metrics; pass "
                    "--include-synthetic for scratch exploration only)"
                )
        data_specs = real_only
    if not data_specs:
        raise DatasetError(
            "no real (non-synthetic) dataset files selected; refusing to train"
        )

    raw = load_dataset(data_specs)
    dataset, dropped = drop_invalid_rows(raw)
    if dropped:
        missing = describe_missing(raw)
        print(
            f"[dataset] dropped {dropped} invalid row(s): "
            f"{missing['columns']}, empty session: {missing['empty_session']}"
        )
    for issue in validate_dataset(dataset):
        print(f"[dataset] {issue}")
    if dataset.size == 0:
        raise DatasetError("cannot train on an empty dataset")

    # -------------------------------------------------------- session split --
    folds = split_sessions(
        dataset,
        test_per_class=args.test_sessions_per_class,
        val_per_class=args.val_sessions_per_class,
        random_state=args.seed,
    )
    print(
        "\nSESSION-BASED SPLIT  (random_state={}, {} test session(s) + "
        "{} validation session(s) per class)".format(
            args.seed, args.test_sessions_per_class, args.val_sessions_per_class
        )
    )
    for fold in ("train", "validation", "test"):
        print(f"  {fold.upper():10s}: {sorted(folds[fold])}")

    seq_len = args.sequence_length

    # ------------------------------ class-balance + feasibility gate --------
    assessment = assess_split(
        dataset,
        folds,
        sequence_length=seq_len,
        min_train_per_class=args.min_train_per_class,
        min_val_per_class=args.min_val_per_class,
        min_test_per_class=args.min_test_per_class,
    )
    _print_class_balance(dataset, folds, assessment, seq_len)

    if not assessment["adequate"]:
        return _stop_insufficient_data(dataset, folds, assessment, args, seq_len)

    # ------------------------------------------------ build windows per fold --
    X_train, y_train, s_train = build_sequences(
        dataset, folds["train"], sequence_length=seq_len
    )
    X_val, y_val, s_val = build_sequences(
        dataset, folds["validation"], sequence_length=seq_len
    )
    X_test, y_test, s_test = build_sequences(
        dataset, folds["test"], sequence_length=seq_len
    )
    for name, (X, y, s) in {
        "TRAIN": (X_train, y_train, s_train),
        "VALIDATION": (X_val, y_val, s_val),
        "TEST": (X_test, y_test, s_test),
    }.items():
        print(
            f"{name:10s}: {X.shape} sequences [{s['n_sequences_alert']} alert / "
            f"{s['n_sequences_drowsy']} drowsy]"
        )

    # ------------------------------------------------------ preprocessing ----
    scaler = fit_scaler(X_train)
    X_train_s = scale_sequences(X_train, scaler)
    X_val_s = scale_sequences(X_val, scaler)
    X_test_s = scale_sequences(X_test, scaler)
    class_weights = compute_class_weights(y_train)
    print(f"class_weights (balanced, from TRAIN): {class_weights}")

    # -------------------------------------------------------------- model ----
    units = tuple(int(u) for u in args.units.split(","))
    model = build_lstm_model(
        seq_len,
        len(FEATURE_COLUMNS),
        units=units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        random_state=args.seed,
    )
    model.summary()

    out = Path(args.out)
    checkpoint = out.parent / f".best.{out.name}"
    _maybe_write(out, args.force)
    _maybe_write(Path(args.scaler_out), args.force)
    _maybe_write(Path(args.metadata_out), args.force)

    import keras

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", save_best_only=True
        ),
    ]
    print(f"\nTraining LSTM (epochs={args.epochs}, batch={args.batch_size}, CPU)...")
    history = model.fit(
        X_train_s,
        y_train,
        validation_data=(X_val_s, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    # Restore the best validation-loss weights.
    if checkpoint.exists():
        model.load_weights(str(checkpoint))

    # ------------------------------------------------------- evaluation ------
    proba_test = model.predict(X_test_s, batch_size=args.batch_size, verbose=0).ravel()
    y_pred_test = (proba_test >= 0.5).astype(int)
    dl_metrics = classification_metrics(y_test, y_pred_test, proba_test)

    print("\nLSTM on unseen TEST sessions:")
    _print_metrics(dl_metrics)

    # ------------------------------- ML reference on the SAME session split ---
    def _frames(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isin(dataset.sessions, ids)
        return dataset.features[mask], (dataset.labels[mask] == "drowsy").astype(int)

    X_train_frames, y_train_frames = _frames(folds["train"])
    X_test_frames, y_test_frames = _frames(folds["test"])
    rf_metrics = _retrain_random_forest(
        X_train_frames, y_train_frames, X_test_frames, y_test_frames
    )
    print("\nRandom Forest (re-trained on the same split) on unseen TEST frames:")
    _print_metrics(rf_metrics)

    # ------------------------------------------------------------ persist ----
    import joblib

    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    joblib.dump(scaler, str(Path(args.scaler_out)))
    if checkpoint.exists():
        checkpoint.unlink()

    preliminary, preliminary_reasons = preliminary_assessment(dataset)
    if preliminary_reasons:
        print(
            "\n[dataset] PRELIMINARY: " + " | ".join(preliminary_reasons)
        )

    metadata = {
        "type": "lstm",
        "status": "COMPLETED",
        "model_path": str(out),
        "scaler_path": str(Path(args.scaler_out)),
        "feature_names": list(FEATURE_COLUMNS),
        "sequence_length": seq_len,
        "architecture": (
            f"LSTM({units[0]}, return_sequences=True) -> Dropout({args.dropout}) "
            f"-> LSTM({units[1]}) -> Dropout({args.dropout}) -> "
            f"Dense({args.dense_units}, relu) -> Dropout({args.dropout}) -> "
            "Dense(1, sigmoid)"
        ),
        "optimizer": "Adam",
        "loss": "binary_crossentropy",
        "preliminary": preliminary,
        "preliminary_reasons": preliminary_reasons,
        "train_session_ids": sorted(folds["train"]),
        "validation_session_ids": sorted(folds["validation"]),
        "test_session_ids": sorted(folds["test"]),
        "train_class_counts": {
            k: int(v)
            for k, v in zip(
                ("alert", "drowsy"),
                (int(np.count_nonzero(y_train == 0)), int(np.count_nonzero(y_train == 1))),
            )
        },
        "validation_class_counts": {
            k: int(v)
            for k, v in zip(
                ("alert", "drowsy"),
                (int(np.count_nonzero(y_val == 0)), int(np.count_nonzero(y_val == 1))),
            )
        },
        "test_class_counts": {
            k: int(v)
            for k, v in zip(
                ("alert", "drowsy"),
                (int(np.count_nonzero(y_test == 0)), int(np.count_nonzero(y_test == 1))),
            )
        },
        "class_weights": class_weights,
        "label_map": {"alert": 0, "drowsy": 1},
        "label_mode": "final",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": args.seed,
        "epochs_run": int(history.epoch[-1]) + 1,
        "test_metrics": dl_metrics,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    Path(args.metadata_out).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nSaved model    -> {out}")
    print(f"Saved scaler   -> {args.scaler_out}")
    print(f"Saved metadata -> {args.metadata_out}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_results(
        results_dir,
        history,
        y_test,
        y_pred_test,
        dl_metrics,
        rf_metrics,
        seq_len,
        preliminary,
        preliminary_reasons,
        assessment,
    )
    print("\nDeep-learning training complete.")
    return 0


def _print_class_balance(
    dataset, folds, assessment, sequence_length,
) -> None:
    """Print the per-class session+sequence balance BEFORE training (task #9)."""
    frame_counts = session_frame_counts(dataset)
    per_class = sessions_by_class(dataset)
    grand_sessions = grand_sequences = grand_frames = 0
    print(
        "\nCLASS-BALANCE REPORT (before training)   "
        f"sequence_length={sequence_length}"
    )
    print(f"  {'class':8s} {'sessions':9s} {'sequences':10s} {'frames':7s}")
    for label in LABELS:
        ids = per_class.get(label, [])
        frames = sum(frame_counts.get(s, 0) for s in ids)
        seqs = sum(
            frame_counts[s] - sequence_length + 1
            for s in ids
            if frame_counts.get(s, 0) >= sequence_length
        )
        grand_sessions += len(ids)
        grand_sequences += seqs
        grand_frames += frames
        print(f"  {label:8s} {len(ids):<9d} {seqs:<10d} {frames:<7d}")
    print(
        f"  {'total':8s} {grand_sessions:<9d} {grand_sequences:<10d} "
        f"{grand_frames:<7d}"
    )
    print("\n  FOLD session assignment and per-fold sequence counts:")
    for fold in ("train", "validation", "test"):
        fold_info = assessment["folds"][fold]
        print(
            f"    {fold.upper():10s} {len(fold_info['session_ids'])} session(s)"
        )
        for label in LABELS:
            entry = fold_info["per_class"][label]
            print(
                f"      {label:8s}: {entry['n_sequences']} sequence(s) from "
                f"{entry['n_sessions']} session(s) {entry['session_ids']}"
            )


def _dataset_summary(dataset, sequence_length: int) -> dict:
    """Rows / sessions / sequences / recording dates for the stopping report."""
    frame_counts = session_frame_counts(dataset)
    per_class = sessions_by_class(dataset)
    summary: dict = {
        "n_rows": dataset.size,
        "n_sessions": len(frame_counts),
        "sequence_length": sequence_length,
        "per_class": {},
        "recording_dates": [],
    }
    for label in LABELS:
        ids = per_class.get(label, [])
        seqs = sum(
            frame_counts[s] - sequence_length + 1
            for s in ids
            if frame_counts.get(s, 0) >= sequence_length
        )
        summary["per_class"][label] = {
            "n_sessions": len(ids),
            "n_sequences": seqs,
            "session_ids": sorted(ids),
            "frames": sum(frame_counts.get(s, 0) for s in ids),
        }
    all_ids: set[str] = set()
    for label in LABELS:
        all_ids.update(summary["per_class"][label]["session_ids"])
    summary["recording_dates"] = sorted(recording_dates(all_ids))
    return summary


def _stop_insufficient_data(
    dataset, folds, assessment, args, sequence_length: int,
) -> int:
    """Feasibility gate failed: write an honest report, DO NOT train.

    The requirement is that a leak-safe session split keeps BOTH classes in
    every fold with enough sequences for the numbers to mean something. When
    that is impossible for the current real data, reporting metrics would be
    misleading (task items #5 and #14), so the script persists the full
    analysis and returns exit code 3 without touching the deployed model.
    """
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    _, prelim_reasons = preliminary_assessment(dataset)
    summary = _dataset_summary(dataset, sequence_length)
    report = {
        "status": "STOPPED_INSUFFICIENT_DATA",
        "message": (
            "Training refused: no deterministic session-based split of the "
            "current real data yields train/validation/test folds that each "
            "keep BOTH classes with enough sequences for meaningful "
            "precision/recall/F1 or a confusion matrix. Reporting numbers "
            "would only reproduce the earlier artefacts (val_accuracy=1.0 "
            "every epoch, roc_auc=1.0 on a one-sample ALERT class) and would "
            "mislead instead of measuring generalization."
        ),
        "reasons": assessment["reasons"],
        "preliminary": True,
        "preliminary_reasons": prelim_reasons,
        "sequence_length": sequence_length,
        "minima": assessment["minima"],
        "dataset_summary": summary,
        "split": assessment["folds"],
        "synthetic_data_excluded": not bool(args.include_synthetic),
        "model_left_untouched": True,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metrics_path = results_dir / "deep_learning_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    txt_path = results_dir / "deep_learning_split_report.txt"
    txt_path.write_text(_stop_report_text(assessment), encoding="utf-8")

    print("\nSTOPPED: refusing to train on the current dataset.")
    for reason in assessment["reasons"]:
        print(f"  - {reason}")
    print(
        "Decision: collect more real sessions, then re-run. A split that "
        "cannot keep both classes with a meaningful per-class count cannot "
        "produce a trustworthy evaluation (task #5/#14)."
    )
    print(f"\nHonest report written to:\n  {metrics_path}\n  {txt_path}")
    print(
        "No model/scaler/metadata was overwritten; the deployed app.py --dl "
        "model stays untouched."
    )
    return 3


def _stop_report_text(assessment) -> str:
    lines = [
        "DEEP-LEARNING TRAINING -- STOPPING REPORT",
        "=========================================",
        "",
        "STATUS: STOPPED_INSUFFICIENT_DATA",
        "",
        "The current real dataset cannot produce a reliable session-based split.",
        "",
        "Sequence minima applied (per class per fold):",
    ]
    for fold, count in assessment["minima"].items():
        lines.append(f"  {fold.upper():10s} >= {count}")
    lines += ["", "Violations:"] + [f"  - {r}" for r in assessment["reasons"]]
    lines += [
        "",
        "A model trained on this split would memorise the training sessions,",
        "show 1.0 validation accuracy on a handful of windows, and report",
        "ROC-AUC=1.0 on a test set too small to contain a meaningful result per",
        "class. Refusing to report numbers is the honest outcome (task #5/#14).",
        "",
        "Collect more real, per-class sessions, then re-run:",
        "  python scripts/train_deep_learning.py --force",
    ]
    return "\n".join(lines)


def _print_metrics(metrics: dict) -> None:
    print(
        f"  accuracy={metrics['accuracy']:.3f}  roc_auc="
        f"{metrics['roc_auc']:.3f}"
    )
    for cls in LABELS:
        m = metrics[cls]
        print(
            f"  {cls:8s} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} (n={m['support']})"
        )
    print(
        f"  macro   P={metrics['macro_precision']:.3f} "
        f"R={metrics['macro_recall']:.3f} F1={metrics['macro_f1']:.3f}"
    )


def _write_results(
    results_dir: Path,
    history,
    y_test,
    y_pred,
    dl_metrics: dict,
    rf_metrics: dict,
    sequence_length: int,
    preliminary: bool,
    preliminary_reasons: list[str],
    assessment: dict,
) -> None:
    import csv

    (results_dir / "deep_learning_metrics.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "metrics": dl_metrics,
                "random_forest_same_split": rf_metrics,
                "history": {
                    "loss": [round(float(v), 5) for v in history.history["loss"]],
                    "val_loss": [round(float(v), 5) for v in history.history["val_loss"]],
                    "accuracy": [round(float(v), 5) for v in history.history["accuracy"]],
                    "val_accuracy": [
                        round(float(v), 5) for v in history.history["val_accuracy"]
                    ],
                },
                "sequence_length": sequence_length,
                "preliminary": preliminary,
                "preliminary_reasons": preliminary_reasons,
                "split": assessment["folds"],
                "split_minima": assessment["minima"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    header = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "drowsy_recall",
        "drowsy_f1",
        "roc_auc",
    ]
    expected = [("random_forest", rf_metrics), ("lstm", dl_metrics)]
    with open(
        results_dir / "deep_learning_model_comparison.csv", "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for name, metrics in expected:
            writer.writerow(
                [
                    name,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['macro_precision']:.4f}",
                    f"{metrics['macro_recall']:.4f}",
                    f"{metrics['macro_f1']:.4f}",
                    f"{metrics['drowsy']['recall']:.4f}",
                    f"{metrics['drowsy']['f1']:.4f}",
                    (
                        f"{metrics['roc_auc']:.4f}"
                        if metrics["roc_auc"] is not None
                        else "n/a"
                    ),
                ]
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=["alert", "drowsy"],
        colorbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion matrix  –  LSTM", fontsize=11)
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix_lstm.png", dpi=120)
    fig.savefig(results_dir / "dl_confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(results_dir / "lstm_training_history.png", dpi=120)
    fig.savefig(results_dir / "dl_training_history.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
