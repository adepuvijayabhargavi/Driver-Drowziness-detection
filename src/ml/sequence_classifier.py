"""Real-time inference wrapper for the temporal LSTM drowsiness classifier.

The wrapper keeps a rolling buffer of the last ``sequence_length`` frame
feature vectors (only frames with a *complete* feature set -- no NaN -- are
pushed), applies the StandardScaler that was fitted on the training data, and
scores the window with a Keras LSTM model saved as ``.keras``.

Behaviour mirrors :class:`src.ml.classifier.DrowsinessClassifier`:

* disabled (with a reason) when the model, scaler or metadata is missing or
  does not match the project's feature schema;
* ``predict()`` returns ``None`` until the buffer is full (warm-up) -- the app
  shows ``DL: WARMING UP`` in that state;
* the verdict is a display/log-only second opinion and never fires alerts.

Keras is imported lazily (the default rule-based run must not pay DL startup
cost). ``KERAS_BACKEND`` defaults to ``torch`` — the CPU-friendly Keras 3
backend installed with ``requirements-dl.txt``.
"""

from __future__ import annotations

import collections
import logging
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.config.settings import MODELS_DIR
from src.ml.schema import FEATURE_COLUMNS, N_FEATURES, validate_feature_order

logger = logging.getLogger(__name__)

DEFAULT_DL_MODEL_PATH = MODELS_DIR / "drowsiness_lstm.keras"
DEFAULT_DL_SCALER_PATH = MODELS_DIR / "drowsiness_lstm_scaler.joblib"
DEFAULT_DL_METADATA_PATH = MODELS_DIR / "drowsiness_lstm_metadata.json"
# Fallback window only used when neither the metadata nor the loaded model
# exposes the trained sequence length.
DEFAULT_DL_SEQUENCE_LENGTH = 30


def _prepare_backend() -> None:
    """Make sure Keras uses the CPU-friendly torch backend when available."""
    os.environ.setdefault("KERAS_BACKEND", "torch")


class DrowsinessSequenceClassifier:
    """Rolling-window temporal classifier over the schema feature vectors."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        sequence_length: int | None = None,
        confidence_threshold: float = 0.85,
    ) -> None:
        self.model_path = (
            Path(model_path) if model_path else DEFAULT_DL_MODEL_PATH
        )
        self.scaler_path = (
            Path(scaler_path) if scaler_path else DEFAULT_DL_SCALER_PATH
        )
        self.metadata_path = (
            Path(metadata_path) if metadata_path else DEFAULT_DL_METADATA_PATH
        )
        # None means "auto-detect": resolved from the trained model's metadata
        # (or its input shape) once the artifacts are loaded.
        self.sequence_length = (
            max(1, int(sequence_length))
            if sequence_length is not None
            else None
        )
        self.confidence_threshold = confidence_threshold

        self._model = None
        self._scaler = None
        self._feature_names: list[str] | None = None
        self._load_reason: str | None = None
        self.buffer: collections.deque[NDArray[np.floating]] = collections.deque(
            maxlen=self.sequence_length or DEFAULT_DL_SEQUENCE_LENGTH
        )

        if self.model_path.exists() and self.scaler_path.exists():
            self._load()
        else:
            missing = [str(p) for p in (self.model_path, self.scaler_path) if not p.exists()]
            message = (
                f"DL model/scaler not found ({', '.join(missing)}) - running "
                "with CV rules only. Train one with `python "
                "scripts/train_deep_learning.py`."
            )
            self._load_reason = message
            logger.warning(message)

    # ------------------------------------------------------------ loading --
    def _load(self) -> None:
        import json

        try:
            _prepare_backend()
            import keras  # noqa: F401  (backend selection happens on import)
            from keras.saving import load_model
        except Exception as exc:  # pragma: no cover - keras missing
            self._load_reason = f"keras/torch not installed - DL disabled ({exc})"
            logger.warning(self._load_reason)
            return

        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            payload = {}
        except Exception as exc:  # pragma: no cover - corrupt metadata
            self._load_reason = f"could not read metadata {self.metadata_path}: {exc}"
            logger.warning(self._load_reason)
            return

        feature_names = [str(n) for n in payload.get("feature_names", [])]
        if feature_names and not validate_feature_order(feature_names):
            self._load_reason = (
                f"{self.metadata_path} uses schema {feature_names} which does "
                "not match the project schema. Refusing to run."
            )
            logger.warning(self._load_reason)
            return

        try:
            model = load_model(str(self.model_path))
            import joblib

            scaler = joblib.load(str(self.scaler_path))
        except Exception as exc:  # pragma: no cover - corrupt artifacts
            self._load_reason = (
                f"could not load DL model/scaler: {exc}"
            )
            logger.warning(self._load_reason)
            return

        inputs = getattr(model, "inputs", None)
        input_shape = getattr(inputs[0], "shape", None) if inputs else None
        expected_seq = (
            int(input_shape[-2])
            if input_shape is not None and len(input_shape) >= 3
            else None
        )
        expected_feat = (
            int(input_shape[-1])
            if input_shape is not None and len(input_shape) >= 1
            else None
        )

        # Resolve the temporal window: the trained length recorded in the
        # model metadata is authoritative; the model input shape is the
        # fallback when the metadata does not expose it.
        metadata_seq = payload.get("sequence_length")
        trained_seq = (
            metadata_seq
            if isinstance(metadata_seq, int) and metadata_seq >= 1
            else expected_seq
        )
        if self.sequence_length is None:
            if trained_seq is None:
                self.sequence_length = DEFAULT_DL_SEQUENCE_LENGTH
                logger.warning(
                    "DL window length not recorded in %s; defaulting to %d.",
                    self.metadata_path,
                    self.sequence_length,
                )
            else:
                self.sequence_length = int(trained_seq)
            self.buffer = collections.deque(maxlen=self.sequence_length)
        elif trained_seq is not None and int(trained_seq) != self.sequence_length:
            raise RuntimeError(
                f"DL sequence-length mismatch: the loaded model was trained "
                f"with a window of {trained_seq} frames but "
                f"--dl-sequence-length was set to {self.sequence_length}. "
                f"Re-run with `--dl-sequence-length {trained_seq}` or omit "
                f"the flag to auto-detect."
            )

        if expected_seq is not None and expected_feat is not None:
            if (
                expected_seq != self.sequence_length
                or expected_feat != N_FEATURES
            ):
                self._load_reason = (
                    f"DL model expects ({expected_seq}, {expected_feat}) "
                    f"inputs but settings say ({self.sequence_length}, "
                    f"{N_FEATURES}). Refusing to run."
                )
                logger.warning(self._load_reason)
                return

        self._model = model
        self._scaler = scaler
        self._feature_names = list(
            feature_names if feature_names else FEATURE_COLUMNS
        )
        logger.info(
            "Loaded DL model from %s (window=%d, %d features)",
            self.model_path,
            self.sequence_length,
            len(self._feature_names),
        )

    # ---------------------------------------------------------- interface --
    @property
    def enabled(self) -> bool:
        return self._model is not None and self._scaler is not None

    @property
    def disabled_reason(self) -> str | None:
        return self._load_reason

    @property
    def feature_names(self) -> list[str] | None:
        return list(self._feature_names) if self._feature_names else None

    @property
    def buffer_length(self) -> int:
        return len(self.buffer)

    @property
    def buffer_ready(self) -> bool:
        return self.enabled and self.buffer_length >= self.sequence_length

    def reset(self) -> None:
        self.buffer.clear()

    def update(self, features: NDArray[np.floating]) -> None:
        """Push a frame feature vector into the rolling buffer (complete only).

        Frames with any NaN (unavailable measurement, e.g. no head pose yet)
        are ignored: a padded/crafted window would silently falsify the input.
        """
        if not self.enabled:
            return
        vector = np.asarray(features, dtype=np.float32).reshape(-1)
        if vector.shape[0] != N_FEATURES or np.isnan(vector).any():
            return
        self.buffer.append(vector)

    def predict(self) -> tuple[str, float] | None:
        """Score the current window; returns (prediction, confidence) or None.

        ``prediction`` is ``"ALERT"`` / ``"DROWSY"``; ``confidence`` is the
        genuine probability of the predicted class (sigmoid output when DROWSY,
        ``1 - sigmoid`` when ALERT). Returns ``None`` until warm-up completes.
        """
        if not self.buffer_ready:
            return None
        window = np.stack(
            [self.buffer[i] for i in range(-self.sequence_length, 0)],
            axis=0,
        )  # (sequence_length, n_features)
        window = self._scaler.transform(window).reshape(
            1, self.sequence_length, N_FEATURES
        ).astype(np.float32)
        proba = float(self._model.predict(window, verbose=0)[0, 0])
        if proba >= 0.5:
            return "DROWSY", proba
        return "ALERT", 1.0 - proba


__all__ = [
    "DrowsinessSequenceClassifier",
    "DEFAULT_DL_MODEL_PATH",
    "DEFAULT_DL_SCALER_PATH",
    "DEFAULT_DL_METADATA_PATH",
    "DEFAULT_DL_SEQUENCE_LENGTH",
]
