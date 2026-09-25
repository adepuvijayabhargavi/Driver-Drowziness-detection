"""Session logger: writes every frame's metrics to a CSV.

The CSV doubles as the labelled dataset for the optional ML classifier
(see ``scripts/train_classifier.py``). ``state`` is written verbatim so you can
filter rows by state afterwards.

Head-pose columns are appended *after* the classic eye/mouth columns so that
training scripts reading only the first columns keep working unchanged.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.dynamics.state_machine import FrameReport

logger = logging.getLogger(__name__)

FIELDS = [
    "frame",
    "timestamp",
    "ear",
    "mar",
    "state",
    "drowsy",
    "yawning",
    "eye_closed_frames",
    "mouth_open_frames",
    "yawn_frames",
    "yawn_duration",
    "yawn_triggered",
    "blink_count",
    "yawn_count",
    "no_face_frames",
    "perclos",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "head_direction",
    "head_pose_duration",
    "head_pose_sustained",
]


class SessionLogger:
    def __init__(self, path: str | Path | None = None) -> None:
        from src.config.settings import SESSIONS_DIR

        if path is None:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            path = SESSIONS_DIR / "session.csv"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        logger.info("Logging session to %s", self.path)

    def write(self, report: FrameReport) -> None:
        self._writer.writerow(
            {
                "frame": report.frame_index,
                "timestamp": round(report.timestamp, 4),
                "ear": "" if report.ear is None else round(report.ear, 4),
                "mar": "" if report.mar is None else round(report.mar, 4),
                "state": report.state.value,
                "drowsy": int(report.drowsy),
                "yawning": int(report.yawning),
                "eye_closed_frames": report.eye_closed_frames,
                "mouth_open_frames": report.mouth_open_frames,
                "yawn_frames": report.yawn_frames,
                "yawn_duration": round(report.yawn_duration, 4),
                "yawn_triggered": int(report.yawn_triggered),
                "blink_count": report.blink_count,
                "yawn_count": report.yawn_count,
                "no_face_frames": report.no_face_frames,
                "perclos": "" if report.perclos is None else round(report.perclos, 2),
                "head_pitch": "" if report.head_pitch is None else round(report.head_pitch, 2),
                "head_yaw": "" if report.head_yaw is None else round(report.head_yaw, 2),
                "head_roll": "" if report.head_roll is None else round(report.head_roll, 2),
                "head_direction": "" if report.head_direction is None else report.head_direction,
                "head_pose_duration": round(report.head_pose_duration, 2),
                "head_pose_sustained": int(report.head_pose_sustained),
            }
        )

    def close(self) -> None:
        self._file.close()


__all__ = ["SessionLogger", "FIELDS"]
