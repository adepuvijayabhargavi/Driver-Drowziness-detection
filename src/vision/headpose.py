"""OpenCV head-pose estimation from 6 facial landmarks.

Estimates the driver's head pitch / yaw / roll every frame using
``cv2.solvePnP``: MediaPipe gives us normalized 2D pixel landmarks, which are
fitted against a canonical rigid 6-point face model. The recovered rotation
matrix is decomposed with ``src.dynamics.headpose.rotation_matrix_to_euler``,
so the *geometry* lives here while the *semantics* / temporal tracking live in
the pure-Python ``src.dynamics`` module (see docs/DESIGN.md).

The camera matrix is approximated with the classic pin-hole shortcut used by
every single-camera head-pose tutorial: focal length (px) == frame width and
principal point == frame center. This is a *soft* approximation -- the
absolute angle accuracy is limited, but the *signs* and relative magnitudes
(the driver turning left vs right, nodding down, etc.) are reliable enough for
a supporting-drowsiness display.
"""

from __future__ import annotations

import logging

import numpy as np

from src.dynamics.headpose import HeadPose, rotation_matrix_to_euler

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh indices used for the head pose, in the same order as the
# rows of ``FACE_MODEL_POINTS_3D`` below:
#     1   = nose tip
#     152 = chin
#     33  = right eye outer corner (subject's right -> left half of the image)
#     263 = left eye outer corner  (subject's left  -> right half of the image)
#     291 = right mouth corner
#     61  = left mouth corner
HEAD_POSE_LANDMARKS: tuple[int, int, int, int, int, int] = (1, 152, 33, 263, 291, 61)

# Canonical rigid face model in millimetres. The origin is the nose tip; the
# +x axis points toward the *subject's left* (so a frontal face aligns with a
# camera looking at it), +y points DOWN the face, +z out of the face toward
# the camera. Values are the widely reused "LearnOpenCV head-pose" dimensions
# with x signs flipped to match MediaPipe's mirrored landmark numbering.
FACE_MODEL_POINTS_3D: np.ndarray = np.array(
    [
        (0.0, 0.0, 0.0),        # 1   nose tip
        (0.0, 63.6, -12.5),     # 152 chin
        (-45.0, -28.5, -9.0),   # 33  right eye corner
        (45.0, -28.5, -9.0),    # 263 left eye corner
        (-22.0, 50.0, -8.0),    # 291 right mouth corner
        (22.0, 50.0, -8.0),     # 61  left mouth corner
    ],
    dtype=np.float32,
)

# Minimum normalized bounding-box width/height of the 6 pose landmarks before a
# solvePnP fit is attempted. Degenerate inputs (all points crushed to one spot,
# missing landmarks reported as (0,0)) would otherwise produce bogus angles.
_MIN_POSE_SPREAD: float = 0.01


def _cv2():
    import cv2  # lazy import: module can be imported without OpenCV installed

    return cv2


class HeadPoseEstimator:
    """Fits the 6-point model to one frame's landmarks and returns Euler angles."""

    def estimate(
        self,
        landmarks: np.ndarray,
        width: int,
        height: int,
        focal: float | None = None,
    ) -> HeadPose | None:
        """Estimate the head pose from *normalized* [0,1]``(N, 2)`` landmarks.

        Returns ``HeadPose`` (pitch, yaw, roll in degrees) or ``None`` when the
        landmarks are unusable (no face, missing pose landmarks, degenerate
        spread) -- the caller should treat ``None`` as "no measurement".
        """
        cv2 = _cv2()
        if landmarks is None or width <= 0 or height <= 0:
            return None
        points = np.asarray(landmarks, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2 or len(points) <= max(HEAD_POSE_LANDMARKS):
            return None
        indices = np.asarray(HEAD_POSE_LANDMARKS, dtype=int)
        if not np.all(np.isfinite(points[indices])):
            return None

        pose2d = points[indices] * np.array([width, height])
        if (
            np.max(pose2d[:, 0]) - np.min(pose2d[:, 0]) < _MIN_POSE_SPREAD * width
            or np.max(pose2d[:, 1]) - np.min(pose2d[:, 1]) < _MIN_POSE_SPREAD * height
        ):
            logger.debug("Head-pose landmarks degenerate (spread too small)")
            return None

        focal = float(width) if focal is None else float(focal)
        camera = np.array(
            [
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros(5, dtype=np.float64)

        try:
            ok, rvec, _ = cv2.solvePnP(
                FACE_MODEL_POINTS_3D,
                np.ascontiguousarray(pose2d, dtype=np.float64),
                camera,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                logger.debug("solvePnP failed to converge for head pose")
                return None
            rotation, _ = cv2.Rodrigues(rvec)
        except cv2.error as exc:
            logger.debug("Head-pose estimation failed: %s", exc)
            return None

        pitch, yaw, roll = rotation_matrix_to_euler(rotation)
        return HeadPose(pitch=pitch, yaw=yaw, roll=roll)


__all__ = [
    "HeadPoseEstimator",
    "HEAD_POSE_LANDMARKS",
    "FACE_MODEL_POINTS_3D",
]
