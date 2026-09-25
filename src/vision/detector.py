"""Face landmark detectors.

Two interchangeable implementations are provided:

1. ``FaceMeshDetector`` (legacy)
   Uses ``mediapipe.solutions.face_mesh``. No model download needed, but the
   ``solutions`` API was **removed in MediaPipe 1.0** -- this backend only works
   on mediapipe < 1.0 (typical for Python 3.10-3.12 installs).

2. ``TasksFaceLandmarkerDetector`` (modern, recommended)
   Uses ``mediapipe.tasks`` FaceLandmarker with the official
   ``face_landmarker.task`` asset. The model is downloaded automatically to
   ``assets/`` on first use (a one-time ~3.5 MB download from Google's model
   store).

``create_detector()`` picks whichever backend actually works on the installed
mediapipe version, so ``python app.py`` just works.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from src.config.settings import ASSETS_DIR

logger = logging.getLogger(__name__)

TASKS_MODEL_PATH = ASSETS_DIR / "face_landmarker.task"
_FACE_MESH_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def _quiet_mediapipe_logging() -> None:
    """Silence MediaPipe's noisy 'Logging before InitGoogle' stderr spam."""
    os.environ.setdefault("GLOG_minloglevel", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# Applied before any mediapipe import happens anywhere in the process.
_quiet_mediapipe_logging()


@dataclass
class FaceData:
    """Normalized facial landmarks + metadata for one frame."""

    landmarks: NDArray[np.floating]        # (N, 2) normalized [0,1] x/y
    face_id: int = 0
    present: bool = True
    meshes: Sequence[object] = field(default_factory=tuple)  # raw backend meshes

    @classmethod
    def empty(cls) -> "FaceData":
        return cls(landmarks=np.zeros((0, 2), dtype=np.float32), present=False)

    def point(self, index: int) -> tuple[float, float]:
        """Normalized (x, y) of one landmark."""
        return float(self.landmarks[index, 0]), float(self.landmarks[index, 1])


class FaceDetector(Protocol):
    """What every detector must provide."""

    def process(self, frame_rgb: NDArray[np.uint8]) -> FaceData:
        """Detect faces; return the first face's landmarks or ``FaceData.empty``."""
        ...

    def close(self) -> None:
        ...


def _legacy_face_mesh_available() -> bool:
    try:
        import mediapipe as mp

        return hasattr(mp, "solutions")
    except ImportError:
        return False


# ============================================================================= #
# Model asset management (the tasks backend needs a .task file).
# ============================================================================= #
def ensure_tasks_model(model_path: str | Path | None = None) -> Path:
    """Make sure the FaceLandmarker model file exists; download it if needed."""
    model_path = Path(model_path) if model_path else TASKS_MODEL_PATH
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading FaceLandmarker model to %s ...", model_path)
    try:
        with urllib.request.urlopen(_FACE_MESH_MODEL_URL) as resp, model_path.open("wb") as out:
            shutil.copyfileobj(resp, out)  # noqa: S310 - pinned https URL
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"Could not download the FaceLandmarker model to {model_path}: {exc}\n"
            "Check your network connection, or download it manually from:\n"
            f"  {_FACE_MESH_MODEL_URL}"
        ) from exc
    logger.info("FaceLandmarker model ready (%d bytes).", model_path.stat().st_size)
    return model_path


# ===================================================================== legacy #
class FaceMeshDetector:
    """Detector backed by the legacy ``mediapipe.solutions.face_mesh`` graph.

    Only works with mediapipe < 1.0. Raises an explicit error on mediapipe >= 1.0
    so the user knows to switch to the tasks backend (or is auto-selected away).
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ImportError(
                "mediapipe is not installed. Run: pip install -r requirements.txt"
            ) from exc
        _quiet_mediapipe_logging()

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This mediapipe version has removed the legacy 'solutions' API. "
                "Use the tasks backend (``--detector tasks``), or install "
                "mediapipe<1.0 for the face_mesh backend."
            )

        self._mp = mp
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.debug("FaceMeshDetector initialised (%s)", getattr(mp, "__version__", "?"))

    def process(self, frame_rgb: NDArray[np.uint8]) -> FaceData:
        results = self._face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return FaceData.empty()

        mesh = results.multi_face_landmarks[0]
        coords = np.array([(lm.x, lm.y) for lm in mesh.landmark], dtype=np.float32)
        return FaceData(landmarks=coords, face_id=0, present=True, meshes=(mesh,))

    def close(self) -> None:
        self._face_mesh.close()


# ======================================================================= tasks #
class TasksFaceLandmarkerDetector:
    """Detector backed by ``mediapipe.tasks`` FaceLandmarker (modern API)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        auto_download: bool = True,
    ) -> None:
        if auto_download:
            model_path = ensure_tasks_model(model_path)
        else:
            model_path = Path(model_path) if model_path else TASKS_MODEL_PATH
            if not model_path.exists():
                raise FileNotFoundError(
                    f"FaceLandmarker model not found at {model_path}. Download it "
                    f"from:\n  {_FACE_MESH_MODEL_URL}\nand place it there."
                )

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "mediapipe (tasks API) is not installed. Run: pip install -r requirements.txt"
            ) from exc
        _quiet_mediapipe_logging()

        self._vision = vision
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._frame_ts = 0
        logger.debug("TasksFaceLandmarkerDetector initialised")

    def process(self, frame_rgb: NDArray[np.uint8]) -> FaceData:
        # The tasks VIDEO mode requires monotonically increasing timestamps.
        self._frame_ts += 1
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=frame_rgb
        )
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)
        if not result.face_landmarks:
            return FaceData.empty()

        lm_list = result.face_landmarks[0]
        coords = np.array([(lm.x, lm.y) for lm in lm_list], dtype=np.float32)
        return FaceData(landmarks=coords, face_id=0, present=True, meshes=(lm_list,))

    def close(self) -> None:
        self._landmarker.close()


# ============================================================================= #
class SyntheticFaceDetector:
    """Deterministic stub used by tests and demos.

    It does not look at the frame; ``process`` simply returns the landmarks
    passed at construction time. Use it to exercise the whole pipeline without
    a webcam.
    """

    def __init__(self, landmarks: NDArray[np.floating] | None = None) -> None:
        # Default: a plausible "alert" face with eyes open and mouth closed.
        self._landmarks = (
            landmarks if landmarks is not None else _open_face_landmarks()
        )

    def process(self, frame_rgb: NDArray[np.uint8]) -> FaceData:
        return FaceData(landmarks=self._landmarks.copy(), present=True)

    def close(self) -> None:
        pass


def _open_face_landmarks() -> NDArray[np.floating]:
    """Build a 478-point skeleton that produces a realistic open-eye EAR
    (~0.3) and a closed-mouth MAR (~0.1). Only eye/mouth landmarks matter."""
    lm = np.zeros((478, 2), dtype=np.float32)
    cx, cy = 0.5, 0.5

    def eye(points: Sequence[int], half_width: float, half_height: float) -> None:
        # p1=(-w,0) p2=(-w/2, h) p3=(w/2, h) p4=(w,0) p5=(w/2,-h) p6=(-w/2,-h)
        # => EAR = h / w for this symmetric construction.
        layout = (
            (-half_width, 0.0),
            (-half_width / 2, half_height),
            (half_width / 2, half_height),
            (half_width, 0.0),
            (half_width / 2, -half_height),
            (-half_width / 2, -half_height),
        )
        for i, (dx, dy) in enumerate(layout):
            lm[points[i]] = (cx + dx, cy + dy)

    eye((33, 160, 158, 133, 153, 144), 0.08, 0.024)      # right eye, EAR=0.3
    eye((362, 385, 387, 263, 373, 380), 0.08, 0.024)     # left eye,  EAR=0.3

    # Closed mouth: vertical gap small, corners far apart. MAR ~ 0.11.
    lm[13] = (cx, cy + 0.020)   # upper inner lip
    lm[14] = (cx, cy - 0.020)   # lower inner lip
    lm[61] = (cx - 0.18, cy)    # left inner corner
    lm[291] = (cx + 0.18, cy)   # right inner corner
    return lm


def create_detector(
    backend: str = "auto",
    max_num_faces: int = 1,
    model_path: str | Path | None = None,
) -> FaceDetector:
    """Factory: build the best available detector.

    ``backend`` in {"auto", "face_mesh", "tasks", "synthetic"}.

    * ``auto``: uses the legacy face_mesh backend when the installed mediapipe
      still ships it (no download needed), otherwise the tasks backend (which
      downloads the model on first use).
    * ``tasks``: always the new MediaPipe Tasks FaceLandmarker.
    """
    backend = (backend or "auto").lower()
    if backend == "synthetic":
        return SyntheticFaceDetector()
    if backend == "face_mesh":
        return FaceMeshDetector(max_num_faces=max_num_faces)
    if backend == "tasks":
        return TasksFaceLandmarkerDetector(
            model_path=model_path or TASKS_MODEL_PATH, num_faces=max_num_faces
        )
    if backend == "auto":
        if _legacy_face_mesh_available():
            logger.info("Using legacy FaceMesh backend (no model download needed).")
            return FaceMeshDetector(max_num_faces=max_num_faces)
        logger.info("Using modern MediaPipe Tasks FaceLandmarker backend.")
        return TasksFaceLandmarkerDetector(
            model_path=model_path or TASKS_MODEL_PATH, num_faces=max_num_faces
        )
    raise ValueError(f"Unknown detector backend: {backend!r} (auto|face_mesh|tasks|synthetic)")


__all__ = [
    "FaceData",
    "FaceDetector",
    "FaceMeshDetector",
    "TasksFaceLandmarkerDetector",
    "SyntheticFaceDetector",
    "create_detector",
    "ensure_tasks_model",
    "TASKS_MODEL_PATH",
]
