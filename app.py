"""Driver Drowsiness Detection and Alert System -- user-facing entry point.

Run from the project root with::

    python app.py

This is the single most convenient way to launch the complete real-time system.
The file deliberately *composes* the whole stack from the modular ``src/``
package so it doubles as a readable architecture overview:

    1. Camera stream            (src.camera.VideoStream)
    2. Face-landmark detector   (src.vision.detector.create_detector)
    3. Audio + visual alerts    (src.alerts.manager.AlertManager)
    4. Detection pipeline       (src.pipeline.DrowsinessPipeline:
                                 EAR/MAR extraction -> temporal state machine
                                 -> optional ML fusion -> alerts)
    5. Live session loop        (src.main.run_session)

Any CLI flag from ``src.main.build_parser`` is supported::

    python app.py --source 0
    python app.py --source path/to/video.mp4 --no-audio
    python app.py --ear-threshold 0.23 --ear-frames 36 --mar-threshold 0.6
    python app.py --ml
"""

from __future__ import annotations

import argparse
import logging

from src.alerts.manager import AlertManager
from src.camera import VideoStream
from src.config.settings import Settings
from src.main import build_parser, build_settings, run_session
from src.pipeline import DrowsinessPipeline
from src.utils.logging_utils import setup_logging
from src.vision.detector import create_detector

logger = logging.getLogger("src")


def build_app_components(args: argparse.Namespace) -> tuple[VideoStream, DrowsinessPipeline]:
    """Instantiate every subsystem in dependency order (camera -> pipeline)."""
    settings: Settings = build_settings(args)

    # 1. Input stream (webcam index or video file).
    stream = VideoStream(settings.camera)

    # 2. MediaPipe face-landmark backend (auto: legacy face_mesh or Tasks).
    detector = create_detector(backend=args.detector)

    # 3. Alert system: loud siren (pygame / winsound) + visual warnings.
    alert_manager = AlertManager(settings=settings.alert)

    # 4. Core pipeline: EAR/MAR + temporal state machine + optional ML + alerts.
    pipeline = DrowsinessPipeline(
        detector=detector,
        settings=settings,
        alert_manager=alert_manager,
        fps=stream.fps,
    )
    return stream, pipeline


def run(argv: list[str] | None = None) -> int:
    """Entry point for ``python app.py``; returns the process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.level)
    logger.info("Driver Drowsiness Detection System starting...")

    try:
        stream, pipeline = build_app_components(args)
    except RuntimeError as exc:
        logger.exception("%s", exc)
        return 1

    run_session(
        stream,
        pipeline,
        fps=stream.fps,
        show_window=not args.no_window,
        save_log=args.save_log,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
