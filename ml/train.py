"""Train, evaluate and save the real ML drowsiness classifier.

Trains three supervised models (Logistic Regression, Random Forest, SVM) on
the schema features collected by ``scripts/collect_ml_data.py``, evaluates
each one on *unseen sessions* (group split, no data leakage), prints an honest
comparison table, saves plots under ``results/`` and exports the best model as
a joblib bundle that the real-time app can load.

Usage::

    python ml/train.py                        # notebook default
    python ml/train.py --data data/features/*.csv --out data/models/my.joblib
    python ml/train.py --force                # overwrite an existing model

The numbers are reported exactly as measured -- the script never fabricates or
inflates accuracy. Treat results as exploratory until your dataset has many
minutes of both classes across multiple drivers.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.typing import NDArray

from src.config.settings import MODELS_DIR, RESULTS_DIR
from src.ml.dataset import (
    Dataset,
    DatasetError,
    describe_missing,
    drop_invalid_rows,
    group_train_test_split,
    load_dataset,
    validate_dataset,
)
from src.ml.schema import FEATURE_COLUMNS, LABELS

logger = logging.getLogger(__name__)

DEFAULT_MODEL_OUT = MODELS_DIR / "drowsiness_ml.joblib"
BUNDLE_SCHEMA_VERSION = 1

# name -> factory used for the automated comparison.
MODEL_FACTORIES: dict[str, object] = {}


def _register_models() -> None:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    MODEL_FACTORIES["logistic_regression"] = (
        lambda seed: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        )
    )
    MODEL_FACTORIES["random_forest"] = (
        lambda seed: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=seed
        )
    )
    # SVC's ``probability=True`` is deprecated (sklearn >= 1.9, removal in
    # 1.11); wrap it in a CalibratedClassifierCV to keep predict_proba.
    MODEL_FACTORIES["svm"] = (
        lambda seed: CalibratedClassifierCV(
            SVC(class_weight="balanced", random_state=seed), ensemble=False
        )
    )


def _vectorize_labels(labels: NDArray[np.str_]) -> NDArray[np.int_]:
    """alert -> 0, drowsy -> 1 (target_names = ['alert', 'drowsy'])."""
    return np.where(labels == "drowsy", 1, 0).astype(np.int_)


def _class_metrics(y_true, y_pred, class_label: str) -> dict[str, float]:
    from sklearn.metrics import precision_recall_fscore_support

    positives = _vectorize_labels(np.asarray([class_label]))[0]
    scores = precision_recall_fscore_support(
        y_true, y_pred, labels=[positives], zero_division=0
    )
    precision, recall, f1, _ = (arr[0] for arr in scores)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(np.count_nonzero(y_true == positives)),
    }


def _metrics_for(y_true, y_pred) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
    )

    result: dict = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for label in LABELS:
        result[label] = _class_metrics(y_true, y_pred, label)
    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=[0, 1], zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=[0, 1], zero_division=0
    )
    result.update(
        {
            "macro_precision": float(macro[0]),
            "macro_recall": float(macro[1]),
            "macro_f1": float(macro[2]),
            "weighted_precision": float(weighted[0]),
            "weighted_recall": float(weighted[1]),
            "weighted_f1": float(weighted[2]),
        }
    )
    return result


def _fit_models(
    X_train,
    y_train,
    X_test,
    y_test,
    random_state: int,
) -> dict[str, dict]:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_train_classes = len(np.unique(y_train))
    if n_train_classes < 2:
        raise DatasetError(
            "the session-based split left a single-class training fold ("
            f"{n_train_classes} class). Rebalance your sessions: record both "
            "alert and drowsy examples from MANY sessions so every split keeps "
            "both classes in the training side."
        )
    if len(np.unique(y_test)) < 2:
        print(
            "[dataset] NOTE: the test fold has a single class; the metric of "
            "the missing class is undefined (reported as 0.0)."
        )

    train_start = time.time()
    results: dict[str, dict] = {}
    for name, factory in MODEL_FACTORIES.items():
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", factory(random_state)),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        entry: dict = {
            "model": model,
            "predictions": y_pred,
            "metrics": _metrics_for(y_test, y_pred),
            "fit_seconds": round(time.time() - train_start, 3),
        }
        inner = model.named_steps.get("clf")
        importance = getattr(inner, "feature_importances_", None)
        if importance is not None:
            entry["feature_importances"] = dict(
                zip(FEATURE_COLUMNS, np.asarray(importance))
            )
        results[name] = entry
    return results


def _choose_best(results: dict[str, dict]) -> str:
    """Best model = highest DROWSY F1; ties broken by macro F1."""
    def score(name: str) -> tuple[float, float]:
        metrics = results[name]["metrics"]
        drowsy_f1 = metrics["drowsy"]["f1"]
        macro_f1 = metrics["macro_f1"]
        return (drowsy_f1, macro_f1)

    best = max(results, key=score)
    return best


def _print_table(results: dict[str, dict]) -> None:
    header = (
        f"{'model':^16s} {'acc':>6s} {'alert P':>8s} {'alert R':>8s} "
        f"{'alert F1':>8s} {'drowsy P':>8s} {'drowsy R':>8s} "
        f"{'drowsy F1':>9s} {'macro P':>8s} {'macro R':>8s} "
        f"{'macro F1':>8s} {'wted F1':>8s}"
    )
    print(header)
    print("-" * len(header))
    for name, entry in results.items():
        m = entry["metrics"]
        print(
            f"{name:16s} {m['accuracy']:6.3f} "
            f"{m['alert']['precision']:8.3f} {m['alert']['recall']:8.3f} "
            f"{m['alert']['f1']:8.3f} {m['drowsy']['precision']:8.3f} "
            f"{m['drowsy']['recall']:8.3f} {m['drowsy']['f1']:9.3f} "
            f"{m['macro_precision']:8.3f} {m['macro_recall']:8.3f} "
            f"{m['macro_f1']:8.3f} {m['weighted_f1']:8.3f}"
        )
    print("-" * len(header))


def _print_feature_importance(entry: dict) -> None:
    """Print Random Forest feature importances ranked by value."""
    ranked = sorted(
        entry["feature_importances"].items(),
        key=lambda item: item[1],
        reverse=True,
    )
    width = max(len(name) for name, _ in ranked)
    print("Feature importance (Random Forest):")
    for name, value in ranked:
        print(f"  {name:<{width}}  {value:.4f}")


def _write_comparison_csv(
    results: dict[str, dict], results_dir: Path
) -> Path:
    """Save a model comparison table (actual measured numbers)."""
    import csv

    header = [
        "model",
        "accuracy",
        "alert_precision",
        "alert_recall",
        "alert_f1",
        "drowsy_precision",
        "drowsy_recall",
        "drowsy_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "model_comparison.csv"
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for name in sorted(results):
            m = results[name]["metrics"]
            writer.writerow(
                [
                    name,
                    f"{m['accuracy']:.4f}",
                    f"{m['alert']['precision']:.4f}",
                    f"{m['alert']['recall']:.4f}",
                    f"{m['alert']['f1']:.4f}",
                    f"{m['drowsy']['precision']:.4f}",
                    f"{m['drowsy']['recall']:.4f}",
                    f"{m['drowsy']['f1']:.4f}",
                    f"{m['macro_precision']:.4f}",
                    f"{m['macro_recall']:.4f}",
                    f"{m['macro_f1']:.4f}",
                    f"{m['weighted_f1']:.4f}",
                ]
            )
    print(f"Saved model comparison to {out}")
    return out


def _write_plots(
    results: dict[str, dict],
    y_test: NDArray[np.int_],
    *feature_names: list[str],
    results_dir: Path,
) -> None:
    """Save confusion-matrix plots + Random Forest feature importance."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError as exc:  # pragma: no cover
        logger.warning("matplotlib not installed; skipping plots (%s)", exc)
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    shared_feature_names = list(feature_names[0]) if feature_names else []

    for name, entry in results.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            entry["predictions"],
            labels=[0, 1],
            display_labels=LABELS,
            colorbar=False,
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(f"Confusion matrix  –  {name}", fontsize=11)
        fig.tight_layout()
        out = results_dir / f"confusion_matrix_{name}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)

    rf_entry = results.get("random_forest")
    if rf_entry is not None and shared_feature_names:
        forest = rf_entry["model"].named_steps["clf"]
        importances = forest.feature_importances_

        fig, ax = plt.subplots(figsize=(8, 5))
        order = np.argsort(importances)
        ax.barh(
            [shared_feature_names[i] for i in order],
            importances[order],
        )
        ax.set_title("Random Forest feature importance")
        ax.set_xlabel("importance")
        fig.tight_layout()
        out = results_dir / "random_forest_feature_importance.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)


def _build_bundle(
    best_name: str,
    results: dict[str, dict],
    dataset: Dataset,
    train_idx: NDArray[np.int_],
    test_idx: NDArray[np.int_],
    random_state: int,
) -> dict:
    train_sessions = {str(s) for s in dataset.sessions[train_idx]}
    test_sessions = {str(s) for s in dataset.sessions[test_idx]}
    from collections import Counter

    train_class_counts = {str(k): int(v) for k, v in Counter(dataset.labels[train_idx]).items()}
    test_class_counts = {str(k): int(v) for k, v in Counter(dataset.labels[test_idx]).items()}
    best_metrics = results[best_name]["metrics"]
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "model": results[best_name]["model"],
        "model_name": best_name,
        "feature_names": list(FEATURE_COLUMNS),
        "preprocessing": {"type": "StandardScaler", "fit_on": "train"},
        "target_names": list(LABELS),
        "label_map": {"alert": 0, "drowsy": 1},
        "metrics": best_metrics,
        "all_models": {
            name: entry["metrics"] for name, entry in results.items()
        },
        "n_samples": int(len(dataset.features)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_sessions_train": len(train_sessions),
        "n_sessions_test": len(test_sessions),
        "train_session_ids": sorted(train_sessions),
        "test_session_ids": sorted(test_sessions),
        "train_class_counts": dict(train_class_counts),
        "test_class_counts": dict(test_class_counts),
        "global_class_counts": {
            str(k): int(v) for k, v in Counter(dataset.labels).items()
        },
        "class_weights": "balanced",
        "sources": list(dataset.sources),
        "random_state": random_state,
        "selection": "best DROWSY F1 (tie: macro F1) on unseen sessions",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_training(args) -> tuple[dict, Path | None]:
    """Train everything and return (report, saved_bundle_path|None)."""
    from collections import Counter

    raw = load_dataset(args.data)
    dataset, dropped = drop_invalid_rows(raw)
    logger.info("dropped %d invalid row(s)", dropped)
    if dropped:
        missing = describe_missing(raw)
        print(
            f"[dataset] dropped {dropped} row(s) that cannot train a "
            "classifier (missing/invalid values, empty session id):"
        )
        for column, count in missing["columns"].items():
            if count and dataset.sources:
                print(f"[dataset]   {column}: {count} missing/invalid value(s)")
        if missing["empty_session"]:
            print(
                f"[dataset]   session_id: {missing['empty_session']} empty "
                "session id(s)"
            )
    issues = validate_dataset(dataset)
    for issue in issues:
        print(f"[dataset] {issue}")

    if dataset.size == 0:
        raise DatasetError("cannot train on an empty dataset")

    _register_models()
    X = np.asarray(dataset.features)
    y = _vectorize_labels(dataset.labels)
    train_idx, test_idx = group_train_test_split(
        dataset,
        test_size=args.test_size,
        random_state=args.seed,
    )

    leak = set(dataset.sessions[train_idx]) & set(dataset.sessions[test_idx])
    if leak:
        logger.warning(
            "LEAKAGE: %d session(s) appear in both splits - results are NOT "
            "a valid generalization estimate",
            len(leak),
        )

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_sessions = sorted(set(dataset.sessions[train_idx]))
    test_sessions = sorted(set(dataset.sessions[test_idx]))

    # ── session breakdown by class ──────────────────────────────────────
    train_label_counts = Counter(dataset.labels[train_idx])
    test_label_counts = Counter(dataset.labels[test_idx])

    print(f"\n{'='*60}")
    print("SESSION-BASED SPLIT  (GroupShuffleSplit, random_state={})".format(
        args.seed
    ))
    print('=' * 60)

    n_all_sessions = len(set(dataset.sessions))
    print(f"\nTRAIN sessions ({len(train_sessions)} of {n_all_sessions}):")
    for cls in sorted(train_label_counts):
        cls_sessions = sorted(
            {
                str(s)
                for s, lb in zip(
                    dataset.sessions[train_idx], dataset.labels[train_idx]
                )
                if lb == cls
            }
        )
        print(f"  {cls:8s}: {cls_sessions}")
    n_alert = train_label_counts.get("alert", 0)
    n_drowsy = train_label_counts.get("drowsy", 0)
    total_train = n_alert + n_drowsy
    print(
        f"  distribution: alert={n_alert} ({n_alert/total_train:.1%}), "
        f"drowsy={n_drowsy} ({n_drowsy/total_train:.1%})"
    )

    print(f"\nTEST sessions ({len(test_sessions)} of {n_all_sessions}):")
    for cls in sorted(test_label_counts):
        cls_sessions = sorted(
            {
                str(s)
                for s, lb in zip(
                    dataset.sessions[test_idx], dataset.labels[test_idx]
                )
                if lb == cls
            }
        )
        print(f"  {cls:8s}: {cls_sessions}")
    tn_alert = test_label_counts.get("alert", 0)
    tn_drowsy = test_label_counts.get("drowsy", 0)
    total_test = tn_alert + tn_drowsy
    print(
        f"  distribution: alert={tn_alert} ({tn_alert/total_test:.1%}), "
        f"drowsy={tn_drowsy} ({tn_drowsy/total_test:.1%})"
    )
    print(
        f"\ntrain rows: {len(X_train)} ({len(train_sessions)} sessions) "
        f"| test rows: {len(X_test)} ({len(test_sessions)} sessions)"
    )
    print('=' * 60 + "\n")

    # ── fit all three models ────────────────────────────────────────────
    results = _fit_models(X_train, y_train, X_test, y_test, args.seed)
    _print_table(results)

    # ── feature importance (Random Forest) ──────────────────────────────
    rf_entry = results.get("random_forest")
    if rf_entry is not None and "feature_importances" in rf_entry:
        print()
        _print_feature_importance(rf_entry)

    best_name = _choose_best(results)
    best_m = results[best_name]["metrics"]
    print(
        f"\nSelected model: {best_name} (best DROWSY F1 on unseen sessions)"
    )
    print(
        f"  DROWSY  recall={best_m['drowsy']['recall']:.3f}  "
        f"f1={best_m['drowsy']['f1']:.3f}  "
        f"macro_f1={best_m['macro_f1']:.3f}"
    )

    # ── plots + CSV ─────────────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    _write_plots(
        results,
        y_test,
        list(FEATURE_COLUMNS),
        results_dir=results_dir,
    )
    _write_comparison_csv(results, results_dir)

    # ── save model bundle ───────────────────────────────────────────────
    out = Path(args.out)
    # Allow {model} placeholder in output path (e.g. drowsiness_real_{model}.joblib)
    if "{model}" in out.name:
        out = out.with_name(out.name.replace("{model}", best_name))

    bundle = _build_bundle(
        best_name, results, dataset, train_idx, test_idx, args.seed
    )
    saved = _save_bundle(bundle, out, args)
    return {
        "best_model": best_name,
        "results": results,
        "metrics": results[best_name]["metrics"],
        "dataset_size": dataset.size,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
        "train_class_counts": dict(train_label_counts),
        "test_class_counts": dict(test_label_counts),
        "leak_sessions": len(leak),
    }, saved


def _save_bundle(bundle: dict, out: Path, args) -> Path | None:
    """Write the joblib bundle, refusing to silently overwrite a model."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        answer = input(
            f"Model {out} already exists. Overwrite it? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("Aborted - not overwriting an existing model.")
            return None
    import joblib

    joblib.dump(bundle, str(out))
    print(f"Saved model bundle to {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        nargs="*",
        default=[str(MODELS_DIR.parent / "features" / "drowsiness_dataset.csv")],
        help="Dataset CSV(s). Glob patterns are expanded.",
    )
    parser.add_argument("--out", default=str(DEFAULT_MODEL_OUT))
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR)
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing model without prompting.",
    )
    args = parser.parse_args(argv)

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
            # Keep the spec so load_dataset can raise a precise error.
            expanded.append(spec)
    # Preserve order and drop duplicates (default path + glob may overlap).
    from collections import OrderedDict

    args.data = list(OrderedDict.fromkeys(expanded))

    report, saved = run_training(args)
    if saved is None:
        return 0
    print("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
