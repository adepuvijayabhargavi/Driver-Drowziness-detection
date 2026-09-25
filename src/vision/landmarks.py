"""Canonical MediaPipe Face Mesh landmark indices.

MediaPipe FaceMesh returns 478 landmarks (468 canonical + 10 iris points when
``refine_landmarks=True``). The indices below follow the documented FaceMesh
topology. They are ordered so that an *aspect-ratio* computation works out of
the box:

    index 0 = outer eye corner
    index 1 = upper eyelid, outer half
    index 2 = upper eyelid, inner half
    index 3 = inner eye corner
    index 4 = lower eyelid, inner half
    index 5 = lower eyelid, outer half

The same 6-point ordering the classic dlib "EAR" papers use
(Soukupova & Cech, 2016).
"""

from __future__ import annotations

from typing import Final, Sequence

# The two 6-point sets used for Eye Aspect Ratio.
# RIGHT_EYE == the subject's right eye (left half of the image, mirrored).
RIGHT_EYE_IDX: Final[Sequence[int]] = (33, 160, 158, 133, 153, 144)
LEFT_EYE_IDX: Final[Sequence[int]] = (362, 385, 387, 263, 373, 380)

# Every eye landmark, useful for drawing the eye outline.
LEFT_EYE_ALL_IDX: Final[Sequence[int]] = (
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    466, 388, 387, 386, 385, 384, 398,
)
RIGHT_EYE_ALL_IDX: Final[Sequence[int]] = (
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
)

# The two points used for the Mouth Aspect Ratio (MAR):
#   13 = center of the upper inner lip
#   14 = center of the lower inner lip
#  61  = inner left mouth corner
#  291 = inner right mouth corner
MOUTH_UPPER_LIP_IDX: Final[Sequence[int]] = (13,)
MOUTH_LOWER_LIP_IDX: Final[Sequence[int]] = (14,)
MOUTH_LEFT_CORNER_IDX: Final[Sequence[int]] = (61,)
MOUTH_RIGHT_CORNER_IDX: Final[Sequence[int]] = (291,)

MOUTH_VERTICAL_IDX: Final[Sequence[int]] = (13, 14)      # distance = mouth openness
MOUTH_HORIZONTAL_IDX: Final[Sequence[int]] = (61, 291)   # distance = mouth width

# Full outer lip outline for rendering.
MOUTH_OUTLINE_IDX: Final[Sequence[int]] = (
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 91,
)

# Every landmark belonging to the two eyes and the mouth (used to draw only
# the *interesting* regions instead of the whole mesh).
ROI_LANDMARKS: Final[Sequence[int]] = tuple(
    sorted(set(LEFT_EYE_ALL_IDX) | set(RIGHT_EYE_ALL_IDX) | set(MOUTH_OUTLINE_IDX))
)

# The 6 landmark indices used by the head-pose estimator, in the same order as
# the rows of ``src.vision.headpose.FACE_MODEL_POINTS_3D``:
#   1   = nose tip, 152 = chin
#   33  = right eye outer corner (subject's right -> left half of the image)
#   263 = left eye outer corner
#   291 = right mouth corner, 61 = left mouth corner
HEAD_POSE_LANDMARKS: Final[Sequence[int]] = (1, 152, 33, 263, 291, 61)

__all__ = [
    "RIGHT_EYE_IDX",
    "LEFT_EYE_IDX",
    "LEFT_EYE_ALL_IDX",
    "RIGHT_EYE_ALL_IDX",
    "MOUTH_UPPER_LIP_IDX",
    "MOUTH_LOWER_LIP_IDX",
    "MOUTH_LEFT_CORNER_IDX",
    "MOUTH_RIGHT_CORNER_IDX",
    "MOUTH_VERTICAL_IDX",
    "MOUTH_HORIZONTAL_IDX",
    "MOUTH_OUTLINE_IDX",
    "ROI_LANDMARKS",
    "HEAD_POSE_LANDMARKS",
]
