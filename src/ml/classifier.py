"""Inference wrapper for the optional ML drowsiness classifier.

Loads a trained-model *bundle* produced by ``ml/train.py`` -- a joblib dict
that embeds the fitted pipeline (model + preprocessing), the ordered feature
schema and the training metrics -- and exposes ``predict`` / ``predict_proba``.
If the model is missing, is a bare legacy estimator (no schema), or carries a
feature schema that does not match ``src.ml.schema``, the wrapper disables
itself with an explanatory message and the pipeline falls back to the classic
CV rules. The real-time app NEVER depends on the ML layer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.config.settings import MODELS_DIR
from src.ml.schema import (
    FEATURE_COLUMNS,
    N_FEATURES,
    validate_feature_order,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = MODELS_DIR / "drowsiness_real_random_forest.joblib"
BUNDLE_SCHEMA_VERSION = 1


class DrowsinessClassifier:
    """Wrapper around a pretrained scikit-learn estimator bundle."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.85,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_name: str | None = None
        self._feature_names: list[str] | None = None
        self._load_reason: str | None = None
        if self.model_path.exists():
            self._load()
        else:
            message = (
                f"ML model not found at {self.model_path} - running with CV "
                "rules only. Train one with `python ml/train.py`."
            )
            self._load_reason = message
            logger.warning(message)

    # ------------------------------------------------------------ loading --
    def _load(self) -> None:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            self._load_reason = f"joblib not installed - ML disabled ({exc})"
            logger.warning(self._load_reason)
            return

        try:
            payload = joblib.load(str(self.model_path))
        except Exception as exc:  # pragma: no cover - corrupt files et al.
            self._load_reason = f"could not load {self.model_path}: {exc}"
            logger.warning(self._load_reason)
            return

        if not isinstance(payload, dict) or not {
            "model",
            "feature_names",
        } <= set(payload):
            self._load_reason = (
                f"{self.model_path} is a bare estimator without a feature "
                "schema (legacy model). Retrain it with `python ml/train.py` "
                "to get a schema-embedded bundle."
            )
            logger.warning(self._load_reason)
            return

        feature_names = [str(n) for n in payload["feature_names"]]
        if not validate_feature_order(feature_names):
            self._load_reason = (
                f"{self.model_path} uses feature schema {feature_names} which "
                f"does not match the project schema {list(FEATURE_COLUMNS)}. "
                "Refusing to run to avoid feeding features in the wrong order."
            )
            logger.warning(self._load_reason)
            return

        self.model = payload["model"]
        self.model_name = str(payload.get("model_name", "unknown"))
        self._feature_names = feature_names
        logger.info(
            "Loaded ML model '%s' from %s (%d features)",
            self.model_name,
            self.model_path,
            len(feature_names),
        )

    # ---------------------------------------------------------- interface --
    @property
    def enabled(self) -> bool:
        return self.model is not None

    @property
    def feature_names(self) -> list[str] | None:
        return list(self._feature_names) if self._feature_names else None

    @property
    def disabled_reason(self) -> str | None:
        return self._load_reason

    def _vector(self, features: NDArray[np.floating]) -> NDArray[np.floating] | None:
        if not self.enabled:
            return None
        vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != N_FEATURES:
            logger.warning(
                "ML feature vector has %d columns but schema requires %d",
                vector.shape[1], N_FEATURES,
            )
            return None
        if np.isnan(vector).any():
            return None
        return vector

    def predict_proba(self, features: NDArray[np.floating]) -> float | None:
        """Probability of the *drowsy* class."""
        vector = self._vector(features)
        if vector is None:
            return None
        proba = self.model.predict_proba(vector)[0]
        drowsy_index = self._drowsy_class_index()
        return float(proba[drowsy_index])

    def predict(self, features: NDArray[np.floating]) -> str | None:
        """Predicted class, normalized to 'ALERT'/'DROWSY'."""
        vector = self._vector(features)
        if vector is None:
            return None
        label = self.model.predict(vector)[0]
        return self._normalize_label(label)

    def evaluate(
        self, features: NDArray[np.floating]
    ) -> tuple[str, float] | None:
        """One-shot (prediction, confidence) inference for a single frame.

        ``confidence`` is the probability of the *predicted* class, so the
        caller can show 'ML Prediction: DROWSY  Confidence: 87.3%' directly.
        Returns ``None`` when the frame cannot be scored (no model, NaN, ...).
        """
        vector = self._vector(features)
        if vector is None:
            return None
        proba = self.model.predict_proba(vector)[0]
        prediction = self.model.predict(vector)[0]
        confidence = self._confidence_for_prediction(prediction, proba)
        return self._normalize_label(prediction), confidence

    # ------------------------------------------------------------ helpers ---
    def _confidence_for_prediction(self, prediction, proba) -> float:
        try:
            classes = list(self.model.classes_)
        except AttributeError:
            return 0.0
        lowered = [str(c).lower() for c in classes]
        pred_low = str(prediction).lower()
        class_index = next(
            (
                i
                for i, c in enumerate(lowered)
                if c == pred_low
                or (pred_low == "1" and str(classes[i]) == "1")
            ),
            None,
        )
        if class_index is None or class_index >= len(proba):
            return 0.0
        return float(proba[class_index])

    def _drowsy_class_index(self) -> int:
        try:
            classes = list(self.model.classes_)
        except AttributeError:
            return 1
        lowered = [str(c).lower() for c in classes]
        if "drowsy" in lowered:
            return lowered.index("drowsy")
        if "1" in [str(c) for c in classes]:
            return [str(c) for c in classes].index("1")
        return len(classes) - 1  # assume the higher index is the positive class

    @staticmethod
    def _normalize_label(label) -> str:
        low = str(label).lower()
        if low in {"1", "drowsy", "true", "positive"}:
            return "DROWSY"
        return "ALERT"


__all__ = ["DrowsinessClassifier", "DEFAULT_MODEL_PATH", "BUNDLE_SCHEMA_VERSION"]
