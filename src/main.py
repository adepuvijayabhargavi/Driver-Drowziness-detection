"""Command-line entry point (module form).

The recommended way to launch the system is the root-level ``app.py``::

    python app.py

The exact same engine can also be launched as a package::

    python -m src

Most of this module is reusable plumbing (CLI parsing, settings construction and
the live session loop) shared with ``app.py``.
"""

from __future__ import annotations

import argparse
import logging
import time

from src.camera import VideoStream
from src.config.settings import CameraConfig, DetectionSettings, Settings
from src.pipeline import DrowsinessPipeline
from src.utils.fps import FPSMeter, sleep_to_fps
from src.utils.logging_utils import setup_logging
from src.utils.session import SessionLogger
from src.vision.detector import create_detector

logger = logging.getLogger("src")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python app.py",
        description="Real-time driver drowsiness detection and alert system (OpenCV + MediaPipe).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default=None,
                   help="Webcam index (0) or path to a video file.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0, help="Target FPS cap.")

    p.add_argument("--detector", default="auto",
                   choices=["auto", "face_mesh", "tasks", "synthetic"],
                   help="Face-landmark backend. 'tasks' auto-downloads the model on first use.")
    p.add_argument("--ear-threshold", type=float, default=None,
                   help="Eye Aspect Ratio below which an eye counts as closed.")
    p.add_argument("--ear-frames", type=int, default=None, dest="ear_consecutive_frames",
                   help="Consecutive frames of closed eyes before a DROWSY alert.")
    p.add_argument("--mar-threshold", type=float, default=None)
    p.add_argument("--mar-frames", type=int, default=None, dest="mar_consecutive_frames")
    p.add_argument("--yawn-duration", type=float, default=None, dest="yawn_duration_seconds",
                   help="Seconds the mouth must stay open to confirm a yawn (FPS-independent).")
    p.add_argument("--recovery-frames", type=int, default=None)

    p.add_argument("--perclos-window", type=float, default=None, dest="perclos_window_seconds",
                   help="Seconds of history used for the rolling PERCLOS window.")
    p.add_argument("--perclos-ear-threshold", type=float, default=None,
                   dest="perclos_ear_threshold",
                   help="EAR below which PERCLOS counts the eyes as closed "
                        "(defaults to --ear-threshold).")
    p.add_argument("--perclos-warning", type=float, default=None, dest="perclos_warning_threshold",
                   help="PERCLOS %% above which an on-screen warning is drawn (no audio).")
    p.add_argument("--perclos-drowsy", type=float, default=None, dest="perclos_drowsy_threshold",
                   help="PERCLOS %% above which (if sustained) contributes to the DROWSY decision.")
    p.add_argument("--perclos-confirm", type=float, default=None, dest="perclos_drowsy_seconds",
                   help="Seconds PERCLOS must stay above --perclos-drowsy before it fires.")
    p.add_argument("--no-perclos", action="store_true",
                   help="Disable the PERCLOS signal entirely (EAR-only behaviour).")

    p.add_argument("--no-head-pose", action="store_true",
                   help="Disable head-pose estimation entirely (EAR/MAR only).")
    p.add_argument("--head-pitch-down", type=float, default=None,
                   dest="head_pitch_down_threshold",
                   help="Pitch (deg, + = down) above which the head counts as nodding down.")
    p.add_argument("--head-pitch-up", type=float, default=None,
                   dest="head_pitch_up_threshold",
                   help="-Pitch (deg) below which the head counts as tipped back.")
    p.add_argument("--head-yaw-left", type=float, default=None,
                   dest="head_yaw_left_threshold",
                   help="Yaw (deg, + = driver's left) above which a left turn registers.")
    p.add_argument("--head-yaw-right", type=float, default=None,
                   dest="head_yaw_right_threshold",
                   help="-Yaw (deg) below which a right turn registers.")
    p.add_argument("--head-duration", type=float, default=None,
                   dest="head_pose_duration_seconds",
                   help="Seconds an abnormal posture must be held before the "
                        "on-screen reminder appears (never audio).")

    p.add_argument("--no-window", action="store_true", help="Disable the OpenCV preview window.")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--save-log", type=str, default=None, metavar="PATH",
                   help="Write frame metrics to a CSV session log.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many frames (handy for video files / CI).")
    p.add_argument("--ml", action="store_true", help="Enable the optional ML classifier fusion.")
    p.add_argument("--ml-model", type=str, default=None, help="Path to a trained .joblib model.")
    p.add_argument("--dl", action="store_true",
                   help="Enable the optional temporal deep-learning (LSTM) classifier.")
    p.add_argument("--dl-model", type=str, default=None,
                   help="Path to a trained .keras temporal model.")
    p.add_argument("--dl-scaler", type=str, default=None,
                   help="Path to the StandardScaler .joblib used by the DL model.")
    p.add_argument("--dl-sequence-length", type=int, default=None,
                   help="Frames per temporal window fed to the DL model "
                        "(default: auto-detected from the trained model).")
    p.add_argument("--level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def overrides_from_args(args: argparse.Namespace) -> dict:
    changes = {}
    for name in (
        "ear_threshold",
        "ear_consecutive_frames",
        "mar_threshold",
        "mar_consecutive_frames",
        "yawn_duration_seconds",
        "recovery_frames",
        "perclos_window_seconds",
        "perclos_ear_threshold",
        "perclos_warning_threshold",
        "perclos_drowsy_threshold",
        "perclos_drowsy_seconds",
        "head_pitch_down_threshold",
        "head_pitch_up_threshold",
        "head_yaw_left_threshold",
        "head_yaw_right_threshold",
        "head_pose_duration_seconds",
    ):
        value = getattr(args, name, None)
        if value is not None:
            changes[name] = value
    return changes


def build_settings(args: argparse.Namespace) -> Settings:
    """Assemble a Settings object from CLI arguments (thresholds, camera, ML)."""
    camera = CameraConfig(
        source=int(args.source) if str(args.source).isdigit() else (args.source or 0),
        width=args.width,
        height=args.height,
        target_fps=args.fps,
    )
    settings = Settings(
        camera=camera,
        detection=DetectionSettings(**overrides_from_args(args)),
    )
    if args.no_audio:
        settings = settings.with_alert(audio_enabled=False)
    if args.no_perclos:
        settings = settings.with_detection(perclos_enabled=False)
    if args.no_head_pose:
        settings = settings.with_detection(head_pose_enabled=False)
    if args.ml:
        settings = settings.with_ml(enabled=True, model_path=args.ml_model)
    if args.dl:
        settings = settings.with_dl(
            enabled=True,
            model_path=args.dl_model,
            scaler_path=args.dl_scaler,
            sequence_length=args.dl_sequence_length,
        )
    return settings


def run_session(
    stream: VideoStream,
    pipeline: DrowsinessPipeline,
    *,
    fps: float,
    show_window: bool = True,
    save_log: str | None = None,
    max_frames: int | None = None,
) -> dict[str, int]:
    """The live loop: read frames, process, display, log -- shared by entrypoints.

    Returns a small statistics dict (frames, drowsy_triggers, yawns_last).
    """
    session = SessionLogger(save_log) if save_log else None
    fps_meter = FPSMeter()
    import cv2

    if show_window:
        cv2.namedWindow("Drowsiness Monitor", cv2.WINDOW_NORMAL)

    stats: dict[str, int] = {"frames": 0, "drowsy_triggers": 0, "yawns_last": 0}
    try:
        for frame in stream:
            frame_start = time.perf_counter()

            annotated, report = pipeline.process_frame(frame)
            stats["frames"] += 1
            stats["yawns_last"] = report.yawn_count
            if report.triggered:
                stats["drowsy_triggers"] += 1

            if session is not None:
                session.write(report)

            if max_frames is not None and stats["frames"] >= max_frames:
                logger.info("Reached --max-frames=%d, stopping.", max_frames)
                break

            current_fps = fps_meter.tick()
            if show_window:
                cv2.putText(
                    annotated,
                    f"{current_fps:4.1f} FPS",
                    (12, annotated.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("Drowsiness Monitor", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    pipeline.reset()
                    logger.info("Session reset")
            sleep_to_fps(fps, frame_start)

            if stats["frames"] % 100 == 0:
                logger.info(
                    "frames=%d drowsy_triggers=%d yawns=%d",
                    stats["frames"], stats["drowsy_triggers"], stats["yawns_last"],
                )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        pipeline.close()
        stream.release()
        if session is not None:
            session.close()
        cv2.destroyAllWindows()

    logger.info(
        "Session ended. Frames: %d, drowsy alerts: %d, yawns: %d, avg FPS: %.1f",
        stats["frames"],
        stats["drowsy_triggers"],
        stats["yawns_last"],
        fps_meter.mean_fps,
    )
    return stats


def run(argv: list[str] | None = None) -> int:
    """Module entry point (``python -m src``)."""
    args = build_parser().parse_args(argv)
    setup_logging(args.level)
    logger.info("Driver Drowsiness Detection System starting...")

    settings = build_settings(args)

    try:
        stream = VideoStream(settings.camera)
    except RuntimeError as exc:
        logger.exception("%s", exc)
        return 1

    detector = create_detector(backend=args.detector)
    pipeline = DrowsinessPipeline(
        detector=detector,
        settings=settings,
        fps=stream.fps,
    )
    run_session(
        stream,
        pipeline,
        fps=stream.fps,
        show_window=not args.no_window,
        save_log=args.save_log,
        max_frames=args.max_frames,
    )
    return 0


__all__ = [
    "build_parser",
    "build_settings",
    "overrides_from_args",
    "run",
    "run_session",
]


if __name__ == "__main__":
    raise SystemExit(run())
