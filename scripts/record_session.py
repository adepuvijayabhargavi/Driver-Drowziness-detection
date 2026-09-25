"""Record a labelled webcam session for ML training.

Produces a CSV with EAR/MAR streams per frame, plus the automatic state label.
You can *correct* the label live with the keyboard:

    d   mark the current frame period as DROWSY
    y   mark as YAWNING
    a   mark as ALERT
    u   unknown (discards the label)

Rows are labelled with the most recent key press. Drop the two bound columns
silently so that downstream scripts can train on them.

Usage::

    python scripts/record_session.py --out data/sessions/me_drowsy.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

from src.camera import VideoStream
from src.config.settings import CameraConfig
from src.dynamics.metrics import average_eye_aspect_ratio, mouth_aspect_ratio
from src.utils.logging_utils import setup_logging
from src.vision.detector import create_detector
from src.vision.landmarks import LEFT_EYE_IDX, RIGHT_EYE_IDX

logger = logging.getLogger("src")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/sessions/session.csv")
    parser.add_argument("--source", default="0")
    parser.add_argument("--detector", default="auto")
    args = parser.parse_args()

    setup_logging("INFO")
    stream = VideoStream(CameraConfig(source=int(args.source)))
    detector = create_detector(args.detector)

    out = open(args.out, "w", newline="")
    writer = csv.writer(out)
    writer.writerow(["timestamp", "ear", "mar", "state"])

    label = "ALERT"
    t0 = time.monotonic()
    print("Press: a=ALERT d=DROWSY y=YAWN u=unknown | q=quit")
    try:
        for frame in stream:
            import cv2

            rgb = frame[:, :, ::-1]
            face = detector.process(rgb)
            now = time.monotonic() - t0
            if face.present:
                ear = average_eye_aspect_ratio(face.landmarks, LEFT_EYE_IDX, RIGHT_EYE_IDX)
                mar = mouth_aspect_ratio(face.landmarks)
                row = (round(now, 3), round(float(ear), 4), round(float(mar), 4), label)
            else:
                row = (round(now, 3), "", "", "NO_FACE")
            writer.writerow(row)

            cv2.imshow("record", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                label = "ALERT"
            elif key == ord("d"):
                label = "DROWSY"
            elif key == ord("y"):
                label = "YAWNING"
            elif key == ord("u"):
                label = "UNKNOWN"
    except KeyboardInterrupt:
        pass
    finally:
        detector.close()
        stream.release()
        out.close()
        import cv2

        cv2.destroyAllWindows()
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
