"""Tests for the runtime classifier (src.ml.classifier)."""

from __future__ import annotations

import numpy as np

from src.ml.classifier import DrowsinessClassifier
from src.ml.schema import FEATURE_COLUMNS, feature_names


def _make_bundle_model(tmp_path, seed=0) -> tuple:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)
    X = rng.rand(120, len(FEATURE_COLUMNS))
    y = (X[:, 2] > 0.5).astype(int)  # ear_mean splits the classes
    model = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    )
    model.fit(X, y)
    bundle = {
        "model": model,
        "model_name": "test_logistic",
        "feature_names": feature_names(),
        "target_names": ["alert", "drowsy"],
        "label_map": {"alert": 0, "drowsy": 1},
        "metrics": {"accuracy": 1.0},
    }
    out = tmp_path / "bundle.joblib"
    joblib.dump(bundle, str(out))
    return out, bundle


def _ready_vector():
    value = np.zeros(len(FEATURE_COLUMNS), dtype=np.float32)
    value[2] = 0.30  # ear_mean
    value[4] = 10.0  # perclos
    return value


class TestBundleLoading:
    def test_loads_valid_bundle(self, tmp_path):
        path, _ = _make_bundle_model(tmp_path)
        clf = DrowsinessClassifier(model_path=path)
        assert clf.enabled
        assert clf.feature_names == feature_names()
        assert clf.model_name == "test_logistic"

    def test_missing_model_disables_with_message(self, tmp_path):
        clf = DrowsinessClassifier(model_path=tmp_path / "does_not_exist.joblib")
        assert not clf.enabled
        assert "not found" in (clf.disabled_reason or "")

    def test_legacy_bare_estimator_disabled(self, tmp_path):
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(0)
        X = rng.rand(40, 10)
        y = (X[:, 2] > 0.5).astype(int)
        model = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))]
        ).fit(X, y)
        out = tmp_path / "legacy.joblib"
        joblib.dump(model, str(out))
        clf = DrowsinessClassifier(model_path=out)
        assert not clf.enabled
        assert "without a feature schema" in (clf.disabled_reason or "")

    def test_feature_order_mismatch_disabled(self, tmp_path):
        import joblib

        _, bundle = _make_bundle_model(tmp_path)
        bundle["feature_names"] = list(reversed(feature_names()))
        out = tmp_path / "reordered.joblib"
        joblib.dump(bundle, str(out))
        clf = DrowsinessClassifier(model_path=out)
        assert not clf.enabled
        assert "wrong order" in (clf.disabled_reason or "")

    def test_corrupt_file_disables_gracefully(self, tmp_path):
        out = tmp_path / "corrupt.joblib"
        out.write_bytes(b"this is definitely not a joblib file")
        clf = DrowsinessClassifier(model_path=out)
        assert not clf.enabled
        assert "could not load" in (clf.disabled_reason or "")


class TestPrediction:
    def test_predict_normalizes_output(self, tmp_path):
        path, _ = _make_bundle_model(tmp_path)
        clf = DrowsinessClassifier(model_path=path)
        label = clf.predict(_ready_vector())
        assert label in ("ALERT", "DROWSY")

    def test_predict_proba_in_unit_interval(self, tmp_path):
        path, _ = _make_bundle_model(tmp_path)
        clf = DrowsinessClassifier(model_path=path)
        prob = clf.predict_proba(_ready_vector())
        assert prob is not None
        assert 0.0 <= prob <= 1.0

    def test_evaluate_returns_verdict_and_confidence(self, tmp_path):
        path, _ = _make_bundle_model(tmp_path)
        clf = DrowsinessClassifier(model_path=path)
        label, confidence = clf.evaluate(_ready_vector())
        assert label in ("ALERT", "DROWSY")
        assert 0.0 <= confidence <= 1.0

    def test_nan_vector_returns_none(self, tmp_path):
        path, _ = _make_bundle_model(tmp_path)
        clf = DrowsinessClassifier(model_path=path)
        vector = _ready_vector()
        vector[2] = np.nan
        assert clf.predict(vector) is None
        assert clf.evaluate(vector) is None
        assert clf.predict_proba(vector) is None

    def test_wrong_vector_width_returns_none(self, tmp_path):
        path, _ = _make_bundle_model(tmp_path)
        clf = DrowsinessClassifier(model_path=path)
        assert clf.predict(np.zeros(5, dtype=np.float32)) is None

    def test_disabled_classifier_returns_none(self, tmp_path):
        clf = DrowsinessClassifier(model_path=tmp_path / "missing.joblib")
        assert clf.predict(_ready_vector()) is None
        assert clf.evaluate(_ready_vector()) is None
