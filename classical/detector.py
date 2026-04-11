from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import config
from classical.features import compute_ear, compute_mar, compute_head_pose


def _ang_diff(angle, baseline):
    """
    Signed angular deviation in (-180, 180], handling wrap-around.
    e.g. _ang_diff(-170, 170) == 20, not -340.
    """
    return ((float(angle) - float(baseline) + 180.0) % 360.0) - 180.0


@dataclass
class FatigueSignal:
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    alerts: List[str] = field(default_factory=list)
    classical_score: float = 0.0
    blink_rate: float = 0.0
    calibrating: bool = False        # True during the head-pose baseline window


_CALIB_FRAMES = 45   # ~1.5 s at 30 fps


class ThresholdDetector:
    """
    Rule-based fatigue detector using EAR, MAR, and head pose thresholds.
    Head pose is measured relative to a personal baseline captured on activation,
    avoiding false positives from natural seated posture offsets.
    No training data required.
    """

    def __init__(self):
        self._eye_closed_frames  = 0
        self._yawn_frames        = 0
        self._head_down_frames   = 0
        self._head_tilt_frames   = 0
        self._blink_count        = 0
        self._frame_count        = 0
        self._was_eye_open       = True

        # Head-pose calibration
        self._calib_pitches: List[float] = []
        self._calib_rolls:   List[float] = []
        self._pitch_baseline = 0.0
        self._roll_baseline  = 0.0
        self._calibrated     = False

    def detect(self, landmarks_px, frame_shape):
        """
        landmarks_px: (468, 2) pixel-coord array from FaceLandmarkExtractor
        Returns FatigueSignal.
        """
        ear   = compute_ear(landmarks_px)
        mar   = compute_mar(landmarks_px)
        pitch, yaw, roll = compute_head_pose(landmarks_px, frame_shape)

        self._frame_count += 1
        alerts = []
        score_components = []

        # ── Head-pose baseline calibration ────────────────────────────────────
        calibrating = False
        if not self._calibrated:
            self._calib_pitches.append(pitch)
            self._calib_rolls.append(roll)
            if len(self._calib_pitches) >= _CALIB_FRAMES:
                self._pitch_baseline = float(np.median(self._calib_pitches))
                self._roll_baseline  = float(np.median(self._calib_rolls))
                self._calibrated     = True
            else:
                calibrating = True

        # Use wrap-safe angular deviation from calibrated baseline
        rel_pitch = _ang_diff(pitch, self._pitch_baseline)
        rel_roll  = _ang_diff(roll,  self._roll_baseline)

        # ── Eye closure ───────────────────────────────────────────────────────
        if ear < config.EAR_THRESHOLD:
            self._eye_closed_frames += 1
            if self._was_eye_open:
                # Transition: open → closed (blink start)
                self._was_eye_open = False
        else:
            if not self._was_eye_open:
                # Transition: closed → open (blink complete)
                self._blink_count += 1
                self._was_eye_open = True
            self._eye_closed_frames = 0

        if self._eye_closed_frames >= config.EAR_CONSEC_FRAMES:
            alerts.append("EYE_CLOSURE")
            score_components.append(1.0)
        else:
            # Partial score based on how close we are to threshold
            score_components.append(self._eye_closed_frames / config.EAR_CONSEC_FRAMES)

        # ── Yawn ──────────────────────────────────────────────────────────────
        if mar > config.MAR_THRESHOLD:
            self._yawn_frames += 1
        else:
            self._yawn_frames = 0

        if self._yawn_frames >= config.MAR_CONSEC_FRAMES:
            alerts.append("YAWN")
            score_components.append(1.0)
        else:
            score_components.append(self._yawn_frames / config.MAR_CONSEC_FRAMES)

        # ── Head nod (pitch) ──────────────────────────────────────────────────
        if not calibrating and abs(rel_pitch) > config.HEAD_PITCH_THRESHOLD:
            self._head_down_frames += 1
        else:
            self._head_down_frames = 0

        if self._head_down_frames >= config.HEAD_CONSEC_FRAMES:
            alerts.append("HEAD_NOD")
            score_components.append(1.0)
        else:
            score_components.append(self._head_down_frames / config.HEAD_CONSEC_FRAMES)

        # ── Head tilt (roll) ──────────────────────────────────────────────────
        if not calibrating and abs(rel_roll) > config.HEAD_ROLL_THRESHOLD:
            self._head_tilt_frames += 1
        else:
            self._head_tilt_frames = 0

        if self._head_tilt_frames >= config.HEAD_CONSEC_FRAMES:
            alerts.append("HEAD_TILT")
            score_components.append(1.0)
        else:
            score_components.append(self._head_tilt_frames / config.HEAD_CONSEC_FRAMES)

        # ── Blink rate ────────────────────────────────────────────────────────
        # Abnormally low blink rate is a drowsiness indicator
        blink_rate = self._blink_count / max(self._frame_count / 30.0, 1.0)

        classical_score = float(sum(score_components) / len(score_components))

        return FatigueSignal(
            ear=ear,
            mar=mar,
            pitch=rel_pitch,
            yaw=yaw,
            roll=rel_roll,
            alerts=alerts,
            classical_score=classical_score,
            blink_rate=blink_rate,
            calibrating=calibrating,
        )

    def reset(self):
        self._eye_closed_frames = 0
        self._yawn_frames       = 0
        self._head_down_frames  = 0
        self._head_tilt_frames  = 0
        self._blink_count       = 0
        self._frame_count       = 0
        self._was_eye_open      = True
        self._calib_pitches     = []
        self._calib_rolls       = []
        self._pitch_baseline    = 0.0
        self._roll_baseline     = 0.0
        self._calibrated        = False
