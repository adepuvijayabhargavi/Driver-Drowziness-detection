"""End-to-end real-time detection pipeline.

Chain:  frame(RGB) -> detect face -> extract landmarks -> EAR/MAR
        -> state machine -> alerts -> annotated BGR frame

The pipeline is deliberately decoupled from any *video source* so the exact
same object can process webcam frames, a video file, offline images or the
test suite's synthetic frames.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from src.alerts.manager import AlertManager
from src.config.settings import Settings
from src.dynamics.headpose import HeadPoseTracker
from src.dynamics.metrics import (
    average_eye_aspect_ratio,
    eye_aspect_ratio,
    mouth_aspect_ratio,
)
from src.dynamics.perclos import PerclosCalculator
from src.dynamics.state_machine import DrowsinessStateMachine, FrameReport
from src.ml.classifier import DrowsinessClassifier
from src.ml.schema import feature_vector_from_report
from src.ml.sequence_classifier import DrowsinessSequenceClassifier
from src.vision.detector import FaceData, FaceDetector
from src.vision.headpose import HeadPoseEstimator
from src.vision.landmarks import LEFT_EYE_IDX, RIGHT_EYE_IDX

logger = logging.getLogger(__name__)


class DrowsinessPipeline:
    """Processes raw frames and returns annotated frames + FrameReports."""

    def __init__(
        self,
        detector: FaceDetector,
        settings: Settings | None = None,
        alert_manager: AlertManager | None = None,
        classifier: DrowsinessClassifier | None = None,
        sequence_classifier: DrowsinessSequenceClassifier | None = None,
        fps: float = 30.0,
    ) -> None:
        self.settings = settings or Settings()
        self.detector = detector
        self.state_machine = DrowsinessStateMachine(
            self.settings.detection, fps=fps
        )
        self.alerts = alert_manager or AlertManager(settings=self.settings.alert)
        self.perclos = PerclosCalculator(
            window_seconds=self.settings.detection.perclos_window_seconds,
            ear_threshold=(
                self.settings.detection.perclos_ear_threshold
                or self.settings.detection.ear_threshold
            ),
        )
        self.head_pose = HeadPoseEstimator()
        self.head_pose_tracker = HeadPoseTracker(
            pitch_down_threshold=self.settings.detection.head_pitch_down_threshold,
            pitch_up_threshold=self.settings.detection.head_pitch_up_threshold,
            yaw_left_threshold=self.settings.detection.head_yaw_left_threshold,
            yaw_right_threshold=self.settings.detection.head_yaw_right_threshold,
            duration_seconds=self.settings.detection.head_pose_duration_seconds,
            smoothing_alpha=self.settings.detection.head_pose_smoothing_alpha,
        )

        self.ml = classifier
        if self.settings.ml.enabled and self.ml is None:
            self.ml = DrowsinessClassifier(
                model_path=self.settings.ml.model_path,
                confidence_threshold=self.settings.ml.confidence_threshold,
            )

        self.dl = sequence_classifier
        if self.settings.dl.enabled and self.dl is None:
            self.dl = DrowsinessSequenceClassifier(
                model_path=self.settings.dl.model_path,
                scaler_path=self.settings.dl.scaler_path,
                sequence_length=self.settings.dl.sequence_length,
                confidence_threshold=self.settings.dl.confidence_threshold,
            )

        self._start_time = time.monotonic()
        self._ml_calls = 0
        self._ml_hits = 0
        self._dl_calls = 0
        self._dl_hits = 0

    # ------------------------------------------------------- frame handling --
    def process_frame(
        self, frame_bgr: np.ndarray, timestamp: float | None = None
    ) -> tuple[np.ndarray, FrameReport]:
        """Run one frame through the whole chain.

        Parameters
        ----------
        frame_bgr : OpenCV BGR image (H, W, 3).
        timestamp : monotonic seconds; falls back to time.monotonic().

        Returns
        -------
        (annotated_bgr_frame, FrameReport)
        """
        if timestamp is None:
            timestamp = time.monotonic() - self._start_time

        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB for MediaPipe
        face = self.detector.process(rgb)

        if not face.present:
            self.perclos.update(None, timestamp)  # advance clock, count nothing
            self._update_head_pose(None, timestamp, report=None)
            report = self.state_machine.update(None, None, timestamp)
            return frame_bgr.copy(), report

        h, w = frame_bgr.shape[:2]
        ear_left = float(eye_aspect_ratio(face.landmarks, LEFT_EYE_IDX))
        ear_right = float(eye_aspect_ratio(face.landmarks, RIGHT_EYE_IDX))
        ear = float(
            average_eye_aspect_ratio(face.landmarks, LEFT_EYE_IDX, RIGHT_EYE_IDX)
        )
        mar = float(mouth_aspect_ratio(face.landmarks))
        perclos_value = None
        if self.settings.detection.perclos_enabled:
            self.perclos.update(ear, timestamp)
            perclos_value = self.perclos.perclos
        report = self.state_machine.update(
            ear, mar, timestamp, perclos=perclos_value
        )
        report.ear_left = ear_left
        report.ear_right = ear_right
        if self.settings.detection.head_pose_enabled:
            self._update_head_pose(face, timestamp, report, w, h)

        # Continuously print the MAR (and EAR) in debug mode so the yawning
        # pipeline can be inspected against the live threshold.
        logger.debug(
            "frame=%d EAR=%.4f MAR=%.4f PERCLOS=%.1f%% | "
            "eyes_closed=%d yawn_frames=%d (%.2fs) | status=%s",
            report.frame_index,
            ear,
            mar,
            0.0 if perclos_value is None else perclos_value,
            report.eye_closed_frames,
            report.yawn_frames,
            report.yawn_duration,
            report.state.name,
        )
        if perclos_value is not None:
            logger.debug(
                "PERCLOS window=%.0fs closed=%.1fs valid=%.1fs perclos=%.1f%%",
                self.settings.detection.perclos_window_seconds,
                self.perclos.closed_time,
                self.perclos.monitoring_time,
                perclos_value,
            )

        # Optional ML fusion: the schema feature vector is scored by the
        # trained classifier (if present). The result is a DISPLAY-only signal
        # fused on top of the reliable CV rules - it never changes the state
        # machine and never fires the siren by itself.
        if self.ml is not None:
            self._ml_calls += 1
            outcome = self.ml.evaluate(feature_vector_from_report(report))
            if outcome is not None:
                prediction, confidence = outcome
                report.ml_prediction = prediction
                report.ml_confidence = confidence
                if (
                    prediction == "DROWSY"
                    and confidence >= self.settings.ml.confidence_threshold
                ):
                    self._ml_hits += 1
                    logger.warning(
                        "ML: %s (%.0f%%) - flagging drowsiness",
                        prediction, confidence * 100,
                    )

        # Optional temporal DL fusion (LSTM): same display-only contract as
        # ML -- a second opinion; never fires the siren.
        if self.dl is not None and self.dl.enabled and report.face_present:
            self._dl_calls += 1
            self.dl.update(feature_vector_from_report(report))
            report.dl_warming_up = not self.dl.buffer_ready
            outcome = self.dl.predict() if self.dl.buffer_ready else None
            if outcome is not None:
                prediction, confidence = outcome
                report.dl_prediction = prediction
                report.dl_confidence = confidence
                if (
                    prediction == "DROWSY"
                    and confidence >= self.settings.dl.confidence_threshold
                ):
                    self._dl_hits += 1
                    logger.warning(
                        "DL: %s (%.0f%%) - flagging drowsiness",
                        prediction, confidence * 100,
                    )
        elif self.dl is not None and not report.face_present:
            self.dl.reset()
            report.dl_warming_up = True

        self.alerts.update(report, timestamp)

        annotated = frame_bgr.copy()
        annotated = self._annotate(annotated, face, report)
        return annotated, report

    # ------------------------------------------------------- head pose aid --
    def _update_head_pose(
        self,
        face: FaceData | None,
        timestamp: float,
        report: FrameReport | None,
        width: int = 0,
        height: int = 0,
    ) -> None:
        """Estimate + track the head pose and attach it to the frame report.

        Head pose is a *supporting* signal: it never changes the state machine
        (the report it mutates is created after ``update()`` ran). When the
        face disappears it feeds ``None`` so the tracker forgets any in-progress
        episode instead of counting the lost-face span as "head held still".
        """
        if face is None or report is None:
            self.head_pose_tracker.feed(None, timestamp)
            return

        pose = self.head_pose.estimate(face.landmarks, width, height)
        tracked = self.head_pose_tracker.feed(pose, timestamp)
        if tracked.direction is not None:
            report.head_pitch = tracked.pitch
            report.head_yaw = tracked.yaw
            report.head_roll = tracked.roll
            report.head_direction = tracked.direction
            report.head_pose_duration = tracked.duration
            report.head_pose_sustained = tracked.sustained
            logger.debug(
                "HEAD pose: %s held %.1fs (pitch=%+6.1f yaw=%+6.1f roll=%+6.1f)",
                tracked.direction,
                tracked.duration,
                tracked.pitch,
                tracked.yaw,
                tracked.roll,
            )

    # -------------------------------------------------------- visual annotate --
    def _annotate(self, frame: np.ndarray, face: FaceData, report: FrameReport):
        from src.alerts.visual import draw_fullscreen_warning
        from src.vision.renderer import (
            COLOR_ORANGE,
            COLOR_RED,
            COLOR_YELLOW,
            draw_face_regions,
            draw_status_overlay,
            state_color,
        )

        _, w = frame.shape[:2]

        if report.face_present:
            frame = draw_face_regions(frame, face.landmarks)

        color = state_color(report.state.name)
        bg = COLOR_RED if report.drowsy else (COLOR_ORANGE if report.yawning else None)
        lines = [f"STATUS: {report.state.value:8s}"]
        lines.append(
            f"EAR: {report.ear:.3f}   Eyes Closed: {report.eye_closed_frames:3d}/"
            f"{self.settings.detection.ear_consecutive_frames}"
        )
        lines.append(
            f"MAR: {report.mar:.3f}   MAR Threshold: "
            f"{self.settings.detection.mar_threshold:.3f}"
        )
        lines.append(
            f"Yawn Frames: {report.yawn_frames:3d}/"
            f"{self.settings.detection.mar_consecutive_frames}"
            f"   ({report.yawn_duration:.2f}s / "
            f"{self.settings.detection.yawn_duration_seconds:.2f}s)"
        )
        perclos_display = 0.0 if report.perclos is None else report.perclos
        lines.append(
            f"PERCLOS: {perclos_display:5.1f}%   "
            f"(Window {self.settings.detection.perclos_window_seconds:.0f}s)"
        )
        if report.head_direction is not None:
            lines.append(
                f"HEAD: {report.head_direction:12s} held {report.head_pose_duration:4.1f}s   "
                f"(P {report.head_pitch:+.0f}  Y {report.head_yaw:+.0f}  "
                f"R {report.head_roll:+.0f})"
            )
        if self.ml is not None:
            if not self.ml.enabled:
                lines.append("ML: ENABLED (no model loaded - CV rules active)")
            elif report.ml_prediction is not None:
                conf = (
                    0.0 if report.ml_confidence is None else report.ml_confidence
                )
                lines.append(
                    f"ML: ENABLED  Prediction: {report.ml_prediction:6s}  "
                    f"Conf: {conf * 100:5.1f}%"
                )
        if self.dl is not None:
            if not self.dl.enabled:
                lines.append("DL: ENABLED (no model loaded - CV rules active)")
            else:
                lines.append(
                    f"DL: ENABLED  DL Window: {self.dl.sequence_length}"
                )
                if report.dl_warming_up:
                    lines.append(
                        f"DL: WARMING UP ({self.dl.buffer_length}/"
                        f"{self.dl.sequence_length})"
                    )
                elif report.dl_prediction is not None:
                    conf = (
                        0.0 if report.dl_confidence is None else report.dl_confidence
                    )
                    lines.append(
                        f"DL: Prediction: {report.dl_prediction:6s}  "
                        f"Conf: {conf * 100:5.1f}%"
                    )
        lines.append(
            f"blinks: {report.blink_count:3d}   yawns: {report.yawn_count:3d}"
        )
        frame = draw_status_overlay(frame, lines, color=color, background=bg)

        # Side hint text.
        frame = draw_status_overlay(
            frame,
            ["q / ESC: quit", "r: reset session"],
            position=(w - 220, 24),
            color=(200, 200, 200),
        )

        if report.drowsy:
            frame = draw_fullscreen_warning(frame, "DROWSINESS ALERT", COLOR_RED)
        elif report.yawning:
            frame = draw_fullscreen_warning(frame, "YAWN DETECTED", COLOR_RED)
        elif (
            report.head_direction is not None and report.head_pose_sustained
        ):
            # Sustained abnormal head posture: on-screen reminder only, with a
            # neutral yellow -- head pose is a supporting signal, never a siren.
            frame = draw_fullscreen_warning(
                frame, f"HEAD {report.head_direction} - CHECK POSTURE", COLOR_YELLOW
            )
        elif (
            report.perclos is not None
            and report.perclos >= self.settings.detection.perclos_warning_threshold
        ):
            # PERCLOS crossed the warning level: on-screen notice only. There
            # is NO audio here -- a single-frame change must never alarm.
            frame = draw_fullscreen_warning(frame, "PERCLOS HIGH - EYES DROOPING", COLOR_ORANGE)
        elif (
            not report.face_present
            and report.no_face_frames > self.settings.detection.no_face_warning_frames
        ):
            frame = draw_fullscreen_warning(frame, "FACE NOT VISIBLE", COLOR_YELLOW)
        return frame

    # ------------------------------------------------------------ lifecycle --
    def reset(self) -> None:
        self.state_machine.reset()
        self.perclos.reset()
        self.head_pose_tracker.reset()
        if self.dl is not None:
            self.dl.reset()

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()
        self.alerts.close()

    @property
    def ml_stats(self) -> tuple[int, int]:
        return self._ml_calls, self._ml_hits

    @property
    def dl_stats(self) -> tuple[int, int]:
        return self._dl_calls, self._dl_hits


__all__ = ["DrowsinessPipeline"]
