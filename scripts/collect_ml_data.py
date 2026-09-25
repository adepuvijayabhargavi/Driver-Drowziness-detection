"""Collect a labelled drowsiness dataset for ML training.

Records one well-formed row of the project ML schema per sampled frame under
the label you declare *beforehand*:

    python scripts/collect_ml_data.py --label drowsy \
        --out data/features/drowsiness_dataset.csv --samples 300

You hold the collection state yourself: sit in a normal driving posture and run
with ``--label alert`` for a while, then act drowsy (slow, prolonged eye
closures, droopy posture...) and run with ``--label drowsy``. Every row keeps
its ``session_id``, and the trainer splits train/test by *whole sessions* so a
model never evaluates on a driver it already memorised.

The preview window is *honest*: while you collect it shows the LIVE camera
frame with the facial landmarks, the current EAR / MAR / PERCLOS / pitch / yaw /
roll read-outs, ``COLLECTING: <LABEL>  Samples: X / N``, and a clear message
-- ``NO FACE DETECTED`` or ``NO CAMERA SIGNAL - BLACK FRAME`` -- instead of a
silent black screen. A sample is written ONLY when a face is present and its
core features (per-eye EAR, MAR) are valid; no-face / black frames are never
counted.

Camera failures are handled explicitly: a frame that ``read()`` cannot deliver
is skipped and reported (never silently treated as a sample), and the capture
stops cleanly if the camera keeps failing.

The camera itself is auto-detected: with no ``--source`` the collector probes
camera indexes 0..N and uses the FIRST one that returns a real image (``read()``
succeeds and the frame is not black/empty) -- ``isOpened()`` alone is never
trusted, because a DirectShow device can "open" and then stream only black
frames. Pass ``--source 0`` / ``--source 1`` (or a video file path) to pin a
specific device.

Sampling
--------
Consecutive webcam frames of one fixed state are near-duplicates, so by
default only every ``--stride``-th frame is saved. The dataset remains a
snapshot of what you actually recorded -- nothing is invented or augmented.

Usage::

    python scripts/collect_ml_data.py --label alert   --samples 300
    python scripts/collect_ml_data.py --label drowsy  --samples 300
    python scripts/collect_ml_data.py --label alert --detector synthetic \
        --no-window --samples 40 --out data/features/demo.csv  # headless check
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import FEATURES_DIR, Settings
from src.ml.schema import FEATURE_COLUMNS, HEADER, LABELS, feature_row
from src.pipeline import DrowsinessPipeline
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_OUT = FEATURES_DIR / "drowsiness_dataset.csv"

WINDOW_NAME = "collect-ml-data"

# A camera warm-up usually yields the first few frames black; only after this
# many CONSECUTIVE black frames do we warn about a no-signal feed.
BLACK_FRAME_GRACE = 5
# Give up after this many consecutive read() failures on a live camera.
MAX_CONSECUTIVE_READ_FAILURES = 60
# How many camera indexes 0..N-1 the auto-scan tries before giving up.
MAX_CAMERA_INDEX_PROBE = 5
# How many frames a candidate camera must yield to be declared "working".
FRAMES_TO_PROBE = 5


class CameraSelectionError(RuntimeError):
    """Raised when no usable camera could be found."""


def new_session_id() -> str:
    """Human-readable yet collision-proof default session identifier.

    Uses second-resolution time plus a random suffix so that two collection
    runs started within the same second (or restarted after a mistake) always
    keep distinct session ids -- the trainer splits by whole session ids and
    would otherwise mix them up.
    """
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _is_effectively_black(frame) -> bool:
    """True when the frame is essentially all-black (no camera signal).

    Costs a cheap mean + dark-pixel fraction per frame; only called from the
    preview path so headless/test runs stay untouched.
    """
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        return False
    return float(np.mean(arr)) < 2.0 and float(np.mean(arr < 25)) > 0.995


def _fmt(value, spec: str = ".2f") -> str:
    if value is None:
        return "?"
    return format(value, spec)


def _feature_overlay(report) -> tuple[str, str]:
    """One-line feature read-outs for the preview (feasible signals only)."""
    metrics = (
        f"EAR {_fmt(report.ear)}  MAR {_fmt(report.mar)}  "
        f"PERCLOS {_fmt(report.perclos, '.1f')}%"
    )
    pose = (
        f"PITCH {_fmt(getattr(report, 'head_pitch', None), '+.0f')}°  "
        f"YAW {_fmt(getattr(report, 'head_yaw', None), '+.0f')}°  "
        f"ROLL {_fmt(getattr(report, 'head_roll', None), '+.0f')}°"
    )
    return metrics, pose


def collect_from_frames(
    frame_iter,
    pipeline: DrowsinessPipeline,
    writer: csv.DictWriter,
    *,
    label: str,
    session_id: str,
    stride: int = 1,
    max_samples: int = 1000,
    max_frames: int | None = None,
    display: bool = False,
) -> int:
    """Save schema rows for ``label`` from a stream of (frame, timestamp).

    One row is written only when the face is present and the core features
    (per-eye EAR and MAR) are finite -- unavailable optional signals stay empty
    in the row and are dropped at training time, never fabricated. Returns the
    number of rows written.
    """
    import cv2  # only needed for the preview path helpers below

    samples_written = 0
    frames_seen = 0
    read_failures = 0
    no_face_frames = 0
    consecutive_black = 0
    warned_black = False

    for item in frame_iter:
        frame, timestamp = item
        frames_seen += 1

        # ------------------------------------------------------------------
        # 1. The capture layer may legitimately return no frame (read() ==
        #    False / None on a flaky webcam). Handle it clearly: skip, report,
        #    continue. Never treat a missing frame as a (black) sample.
        if frame is None:
            read_failures += 1
            logger.warning(
                "collector: read() returned no frame (#%d); skipping",
                read_failures,
            )
            if max_frames is not None and frames_seen >= max_frames:
                break
            continue

        # 2. Run the frame through the REAL pipeline (same one app.py uses):
        #    MediaPipe face tracking -> EAR/MAR/PERCLOS/head pose -> state
        #    machine -> annotated frame. The preview shows `annotated`, which
        #    is a copy of THIS frame plus overlays -- never replaced by a
        #    black/empty canvas.
        annotated, report = pipeline.process_frame(frame, timestamp=timestamp)

        # 3. Sample counting: valid face + valid core features ONLY. A frame
        #    with no face, or a black feed with no face, is never counted.
        core_ok = (
            report.face_present
            and report.ear is not None
            and report.ear_left is not None
            and report.ear_right is not None
            and report.mar is not None
        )
        if core_ok and (max_samples <= 0 or samples_written < max_samples):
            if frames_seen % max(1, stride) == 0:
                try:
                    row = feature_row(
                        report,
                        timestamp=timestamp,
                        session_id=session_id,
                        label=label,
                    )
                except Exception as exc:  # pragma: no cover - defensively loud
                    logger.error(
                        "feature extraction FAILED for frame %d: %s "
                        "(row NOT written)", frames_seen, exc,
                    )
                    raise
                writer.writerow(row)
                samples_written += 1
                missing = [c for c in FEATURE_COLUMNS if row[c] == ""]
                if missing:
                    logger.info(
                        "sample %d written; missing optional feature(s): %s",
                        samples_written, ", ".join(missing),
                    )
                if samples_written % 25 == 0 or samples_written == 1:
                    print(f"   Samples collected: {samples_written}  ({label})")
        elif not report.face_present:
            no_face_frames += 1

        # 4. Live preview: real landmarks are drawn by the pipeline; the
        #    collector adds its status + feature read-outs. The PRIMARY message
        #    is the camera status -- "NO FACE DETECTED" is shown only when the
        #    camera IS delivering frames but no face is visible, so a dead
        #    feed never masks itself as a face-detection problem.
        if display:
            if _is_effectively_black(frame):
                consecutive_black += 1
            else:
                consecutive_black = 0
            no_signal = consecutive_black >= BLACK_FRAME_GRACE

            target = f"{max_samples}" if max_samples > 0 else "unlimited"
            status = (
                f"COLLECTING: {label.upper()}  Samples: {samples_written} / "
                f"{target}  Stride: {stride}"
            )
            color = (0, 255, 0) if label == "alert" else (0, 165, 255)

            if no_signal:
                import numpy as np

                mean = float(np.mean(frame))
                cv2.putText(
                    annotated, "NO CAMERA SIGNAL - BLACK FRAME", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    "The webcam opened but delivers no visible image. Close "
                    "apps using it (Teams/Zoom/OBS/Windows Hello),",
                    (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    "then re-run. You can force another device with --source N.",
                    (20, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    annotated, status, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA,
                )
                if not warned_black:
                    warned_black = True
                    logger.warning(
                        "webcam feed looks BLACK/no-signal (frame mean=%.3f) "
                        "for %d consecutive frames. No face can be detected on "
                        "a black feed and no samples will be collected. Check "
                        "that the camera is free and re-run.", mean, consecutive_black,
                    )
            else:
                cv2.putText(
                    annotated, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
                )
                if report.face_present:
                    metrics, pose = _feature_overlay(report)
                    cv2.putText(
                        annotated, metrics, (20, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 1, cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated, pose, (20, 104),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 1, cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        annotated, "NO FACE DETECTED", (20, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA,
                    )

            cv2.imshow(WINDOW_NAME, annotated)  # noqa: S110 - display helper
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                logger.info("collector stopped by user (q/ESC)")
                break
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("collector stopped (preview window closed)")
                    break
            except cv2.error:
                pass

        # 5. Stopping rules (unbounded recording honours max_samples <= 0
        #    as "until the user stops it").
        if max_samples > 0 and samples_written >= max_samples:
            break
        if max_frames is not None and frames_seen >= max_frames:
            break

    if read_failures:
        logger.warning(
            "collector finished with %d skipped read failure(s)", read_failures
        )
    if display:
        cv2.destroyAllWindows()
    return samples_written


def _iter_synthetic_frames(
    count: int, fps: float, width: int, height: int
):
    """Blank frames fed to a synthetic detector (webcam-free demos/tests)."""
    import numpy as np

    for i in range(count):
        yield np.zeros((height, width, 3), dtype=np.uint8), i / max(fps, 1e-3)


def select_camera_source(
    preferred: int | str | None,
    *,
    probe,
    max_index: int = MAX_CAMERA_INDEX_PROBE,
    status=print,
):
    """Pick the first camera/source that actually delivers real frames.

    ``probe(source)`` must return ``(ok, stream, reason)``. With no
    ``preferred`` source the scan walks camera indexes 0..max_index-1 and
    returns the first one that works; an explicitly requested ``--source`` is
    probed alone. Raises :class:`CameraSelectionError` listing what was tried.
    """
    if preferred is None:
        candidates = list(range(max_index))
        is_index = True
    else:
        candidates = [preferred]
        is_index = isinstance(preferred, int) or str(preferred).isdigit()

    for candidate in candidates:
        label = f"camera {candidate}" if is_index else f"source {candidate}"
        status(f"Testing {label}...")
        ok, stream, reason = probe(candidate)
        if ok:
            status(f"{label.capitalize()}: working")
            role = "camera source" if is_index else "video source"
            status(f"Using {role}: {candidate}")
            return candidate, stream
        status(f"{label.capitalize()}: no signal ({reason})")

    raise CameraSelectionError(
        "No working camera found.\n"
        f"Camera indexes tested: {sorted(set(candidates))}\n"
        "The webcam may be in use by another application (Teams/Zoom/OBS/"
        "Windows Hello) or unavailable. Close it and re-run, or try "
        "`--source N` with a specific camera index."
    )


def _probe_camera(
    source_value,
    *,
    width: int,
    height: int,
    require_live: bool,
) -> tuple[bool, object, str]:
    """Open a camera (or video file) and REQUIRE a usable frame from it.

    ``isOpened()`` alone is not enough -- a DirectShow camera frequently
    "opens" but then streams nothing but black/empty frames. This reads several
    frames and only calls the source "working" when at least one frame is a
    real image (``ret == True``, frame not None and not effectively black).
    ``require_live=False`` is used for video files, where any readable frame is
    fine (clips may legitimately be dark).
    """
    from src.camera import VideoStream
    from src.config.settings import CameraConfig

    try:
        stream = VideoStream(
            CameraConfig(source=source_value, width=width, height=height)
        )
    except RuntimeError as exc:
        return False, None, f"could not open: {exc}"

    last_frame = None
    for _ in range(FRAMES_TO_PROBE):
        last_frame = stream.read()
        if last_frame is None:
            break
        if not require_live or not _is_effectively_black(last_frame):
            return True, stream, "working"

    stream.release()
    reason = (
        "read() returned no frames"
        if last_frame is None
        else "only black/empty frames"
    )
    return False, None, reason


def _frames_from_stream(stream, *, is_camera: bool):
    """Yield (frame, timestamp) from an already-open stream.

    A live camera may drop a frame now and then; a ``read()`` that returns
    None is forwarded as ``(None, timestamp)`` so the caller can skip it,
    and the stream keeps going unless too many consecutive reads fail. A video
    file stops at end-of-file.
    """
    t0 = time.monotonic()
    consecutive_failures = 0
    while True:
        frame = stream.read()
        if frame is None:
            consecutive_failures += 1
            if not is_camera or consecutive_failures > MAX_CONSECUTIVE_READ_FAILURES:
                if not is_camera:
                    logger.info("video source ended")
                else:
                    logger.warning(
                        "camera failed to deliver %d consecutive frames; "
                        "stopping capture", consecutive_failures,
                    )
                break
        else:
            consecutive_failures = 0
        yield frame, time.monotonic() - t0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        required=True,
        choices=LABELS,
        help="Ground truth of the WHOLE recording session.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Target samples for this session (0 = until stopped).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Save every N-th frame to avoid near-duplicate consecutive rows.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Camera index (0, 1, ...) or a video file path. Default: auto-scan "
            "camera indexes 0..N and use the first that delivers real frames."
        ),
    )
    parser.add_argument("--detector", default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many processed frames (headless runs).",
    )
    args = parser.parse_args(argv)
    setup_logging("INFO")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    append = out_path.exists()
    label = args.label.lower()
    session_id = args.session_id or new_session_id()
    display = not args.no_window

    settings = Settings().with_alert(audio_enabled=False)
    stream = None
    if args.detector == "synthetic":
        from src.vision.detector import SyntheticFaceDetector

        detector = SyntheticFaceDetector()
        pipeline_fps = args.fps
        frame_iter = _iter_synthetic_frames(
            args.max_frames or max(args.samples * args.stride + 32, 64),
            args.fps,
            args.width,
            args.height,
        )
    else:
        from src.vision.detector import create_detector

        detector = create_detector(args.detector)

        preferred: int | str | None = None
        if args.source is not None:
            preferred = (
                int(args.source)
                if str(args.source).isdigit()
                else args.source
            )
        try:
            source_value, stream = select_camera_source(
                preferred,
                probe=lambda src: _probe_camera(
                    src,
                    width=args.width,
                    height=args.height,
                    require_live=(
                        preferred is None
                        or isinstance(src, int)
                        or str(src).isdigit()
                    ),
                ),
            )
        except CameraSelectionError as exc:
            print(exc)
            return 1
        is_camera = (
            preferred is None
            or isinstance(source_value, int)
            or str(source_value).isdigit()
        )
        pipeline_fps = stream.fps
        frame_iter = _frames_from_stream(stream, is_camera=is_camera)

    pipeline = DrowsinessPipeline(
        detector=detector, settings=settings, fps=pipeline_fps
    )

    with open(out_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HEADER))
        if not append:
            writer.writeheader()
        else:
            print(f"Appending session to existing dataset {out_path}")

        print(
            f"COLLECTING: {label.upper()}  session={session_id}  "
            f"target={args.samples} samples  stride={args.stride}"
        )
        print("Live preview:  q/Esc to stop early")
        try:
            collected = collect_from_frames(
                frame_iter,
                pipeline,
                writer,
                label=label,
                session_id=session_id,
                stride=args.stride,
                max_samples=args.samples,
                max_frames=args.max_frames,
                display=display,
            )
        except KeyboardInterrupt:
            collected = 0
        finally:
            pipeline.close()
            if stream is not None:
                stream.release()

    print(f"Samples collected: {collected} for label {label}")
    print(f"Dataset written to {out_path}")
    if out_path.exists():
        with out_path.open("r", newline="", encoding="utf-8") as chk:
            n = sum(1 for _ in chk) - 1
        print(f"Total rows in dataset (minus header): {max(n, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
