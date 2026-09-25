"""End-to-end tests for ml/train.py (tiny synthetic datasets, no webcam)."""

from __future__ import annotations

import csv
import types

import numpy as np
import pytest

from ml.train import run_training
from src.ml.classifier import DrowsinessClassifier
from src.ml.schema import FEATURE_COLUMNS, HEADER


def _separable_rows(rng: np.random.RandomState) -> list[dict]:
    """8 sessions (4 alert / 4 drowsy), strongly separable, small noise."""
    rows = []
    idx = 0
    for session in range(1, 9):
        drowsy = session % 2 == 0
        base_ear = 0.31 if not drowsy else 0.13
        base_duration = 0.05 if not drowsy else 3.2
        base_perclos = 8.0 if not drowsy else 55.0
        for _ in range(25):
            jitter = rng.uniform(-0.01, 0.01)
            ear = base_ear + jitter
            rows.append(
                {
                    "timestamp": str(idx),
                    "session_id": f"session-{session}",
                    "ear_left": f"{ear:.4f}",
                    "ear_right": f"{ear:.4f}",
                    "ear_mean": f"{ear:.4f}",
                    "mar": "0.08" if not drowsy else "0.45",
                    "perclos": f"{base_perclos + rng.uniform(-3, 3):.1f}",
                    "eye_closed_duration": f"{base_duration:.3f}",
                    "yawn_duration": ("0.0" if not drowsy else "1.4"),
                    "pitch": ("1.0" if not drowsy else "-12.0"),
                    "yaw": "-2.0",
                    "roll": "1.0",
                    "label": "drowsy" if drowsy else "alert",
                }
            )
            idx += 1
    return rows


def _write_dataset(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HEADER))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def training_args(tmp_path):
    dataset = tmp_path / "train.csv"
    _write_dataset(dataset, _separable_rows(np.random.RandomState(42)))
    return types.SimpleNamespace(
        data=[str(dataset)],
        out=str(tmp_path / "bundle.joblib"),
        results_dir=str(tmp_path / "results"),
        test_size=0.25,
        seed=0,
        force=True,
    )


class TestEndToEndTraining:
    def test_runs_and_saves_bundle(self, training_args):
        report, saved = run_training(training_args)
        assert saved is not None
        assert saved.exists()
        assert report["best_model"] in {
            "logistic_regression",
            "random_forest",
            "svm",
        }
        metrics = report["metrics"]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        for label in ("alert", "drowsy"):
            for key in ("precision", "recall", "f1", "support"):
                assert key in metrics[label]

    def test_no_train_test_session_leakage(self, training_args):
        import joblib

        run_training(training_args)
        bundle = joblib.load(training_args.out)
        assert bundle["n_sessions_train"] + bundle["n_sessions_test"] == 8
        assert bundle["n_sessions_test"] >= 1

    def test_bundle_embeds_schema_and_reloads(self, training_args):
        run_training(training_args)
        clf = DrowsinessClassifier(model_path=training_args.out)
        assert clf.enabled
        assert clf.feature_names == list(FEATURE_COLUMNS)
        vector = np.asarray(
            [0.30, 0.30, 0.30, 0.09, 9.0, 0.0, 0.0, 1.0, -2.0, 1.0],
            dtype=np.float32,
        )
        label, confidence = clf.evaluate(vector)
        assert label in ("ALERT", "DROWSY")
        assert 0.0 <= confidence <= 1.0


class TestPlots:
    def test_confusion_and_importance_plots_written(self, training_args):
        from pathlib import Path

        pytest.importorskip("matplotlib")
        run_training(training_args)
        results_dir = Path(training_args.results_dir)
        for name in ("logistic_regression", "random_forest", "svm"):
            assert (results_dir / f"confusion_matrix_{name}.png").exists()
        assert (results_dir / "random_forest_feature_importance.png").exists()

    def test_drop_report_prints_removed_rows(self, training_args, capsys):
        import csv as _csv
        from pathlib import Path

        path = Path(training_args.data[0])
        rows = list(_csv.DictReader(open(path, encoding="utf-8")))
        rows[3]["ear_left"] = ""
        rows[5]["session_id"] = ""
        _write_dataset(path, rows)

        report, saved = run_training(training_args)
        out = capsys.readouterr().out
        assert saved is not None
        assert "dropped 2 row(s)" in out
        assert "ear_left: 1 missing/invalid value(s)" in out
        assert "session_id: 1 empty session id(s)" in out
        assert report["dataset_size"] == len(rows) - 2


class TestImbalanceAndReporting:
    def test_class_weight_balanced_in_all_models(self):
        from ml.train import MODEL_FACTORIES, _register_models

        _register_models()
        for name, factory in MODEL_FACTORIES.items():
            clf = factory(0)
            # CalibratedClassifierCV exposes the wrapped SVC's params via
            # the 'estimator__class_weight' key; direct estimators use
            # 'class_weight'. RandomForest also has an `.estimator` attr (the
            # base tree), so rely on the get_params() keys, not hasattr().
            params = clf.get_params()
            value = params.get("class_weight")
            if value is None:
                value = params.get("estimator__class_weight")
            assert value == "balanced", name

    def test_metrics_include_macro_and_weighted(self, training_args):
        report, _ = run_training(training_args)
        metrics = report["metrics"]
        for key in (
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        ):
            assert isinstance(metrics[key], float), key
            assert 0.0 <= metrics[key] <= 1.0, key

    def test_bundle_records_sessions_and_class_counts(self, training_args):
        import joblib

        run_training(training_args)
        bundle = joblib.load(training_args.out)
        train_ids = set(bundle["train_session_ids"])
        test_ids = set(bundle["test_session_ids"])
        assert train_ids and test_ids
        assert train_ids.isdisjoint(test_ids)
        assert set(bundle["train_class_counts"]) <= {"alert", "drowsy"}
        assert set(bundle["test_class_counts"]) <= {"alert", "drowsy"}
        assert bundle["class_weights"] == "balanced"
        assert set(bundle["global_class_counts"]) <= {"alert", "drowsy"}

    def test_comparison_csv_written_for_all_models(self, training_args):
        import csv as _csv
        from pathlib import Path

        run_training(training_args)
        path = Path(training_args.results_dir) / "model_comparison.csv"
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(_csv.DictReader(handle))
        assert len(rows) == 3
        assert {r["model"] for r in rows} == {
            "logistic_regression",
            "random_forest",
            "svm",
        }
        first = rows[0]
        assert {"accuracy", "drowsy_recall", "drowsy_f1", "macro_f1"} <= set(
            first
        )

    def test_output_placeholder_expanded_to_best_model(self, tmp_path):
        dataset = tmp_path / "train.csv"
        _write_dataset(dataset, _separable_rows(np.random.RandomState(1)))
        args = types.SimpleNamespace(
            data=[str(dataset)],
            out=str(tmp_path / "drowsiness_real_{model}.joblib"),
            results_dir=str(tmp_path / "results"),
            test_size=0.25,
            seed=0,
            force=True,
        )
        report, saved = run_training(args)
        assert saved is not None
        assert saved.name == f"drowsiness_real_{report['best_model']}.joblib"
        assert saved.exists()

    def test_feature_importance_and_session_split_printed(
        self, training_args, capsys
    ):
        run_training(training_args)
        out = capsys.readouterr().out
        assert "Feature importance (Random Forest):" in out
        assert "perclos" in out
        assert "TRAIN sessions" in out
        assert "TEST sessions" in out
        assert "GroupShuffleSplit" in out
        assert "distribution: alert=" in out and "drowsy=" in out

    def test_cli_accepts_absolute_dataset_path(self, training_args, tmp_path):
        from ml.train import main

        out_path = tmp_path / "cli_bundle.joblib"
        exit_code = main(
            [
                "--data",
                training_args.data[0],
                "--out",
                str(out_path),
                "--force",
                "--seed",
                "0",
            ]
        )
        assert exit_code == 0
        assert out_path.exists()
