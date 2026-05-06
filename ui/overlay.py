import cv2
import numpy as np
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "vision_group_matplotlib"),
)

import mediapipe as mp
import pygame
from classical.landmarks import LEFT_EYE, RIGHT_EYE

mp_drawing       = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands         = mp.solutions.hands

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
_C_GREEN    = (100, 220, 100)
_C_RED      = (70,  70,  230)
_C_CYAN     = (220, 200,  50)
_C_WHITE    = (235, 235, 235)
_C_DIM      = (130, 130, 130)
_C_BLACK    = (  0,   0,   0)
_C_PANEL    = ( 18,  18,  18)
_C_EYE      = (200, 230,  50)   # eye landmark colour
_C_CALIB    = (50,  200, 200)   # calibration progress colour

# ── Typography ────────────────────────────────────────────────────────────────
_F  = cv2.FONT_HERSHEY_DUPLEX   # cleaner than SIMPLEX
_AA = cv2.LINE_AA


def _txt(frame, text, pos, scale, colour, thickness=1):
    """Antialiased text with a dark drop-shadow for legibility."""
    cv2.putText(frame, text, (pos[0] + 1, pos[1] + 1),
                _F, scale, _C_BLACK, thickness + 1, _AA)
    cv2.putText(frame, text, pos, _F, scale, colour, thickness, _AA)


def _panel(frame, x1, y1, x2, y2, alpha=0.55, colour=_C_PANEL, radius=6):
    """Draw a rounded semi-transparent filled rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Subtle border
    cv2.rectangle(frame, (x1, y1), (x2, y2), _C_DIM, 1, _AA)


def _bar(frame, x1, y1, x2, y2, fraction, fg_colour, bg_colour=_C_PANEL):
    """Horizontal progress bar."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_colour, -1)
    fill_x = int(x1 + (x2 - x1) * max(0.0, min(1.0, fraction)))
    if fill_x > x1:
        cv2.rectangle(frame, (x1, y1), (fill_x, y2), fg_colour, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), _C_DIM, 1, _AA)


# ── Eye landmark drawing ──────────────────────────────────────────────────────

def draw_eye_landmarks(frame, landmarks_px):
    """
    Draw eye contours and landmark dots on the frame.
    landmarks_px: (468, 2) float32 pixel coordinates.
    """
    for indices in (LEFT_EYE, RIGHT_EYE):
        pts = landmarks_px[indices].astype(np.int32)
        # Draw contour polygon
        cv2.polylines(frame, [pts.reshape(-1, 1, 2)], isClosed=True,
                      color=_C_EYE, thickness=1, lineType=_AA)
        # Draw each landmark dot
        for pt in pts:
            cv2.circle(frame, tuple(pt), 2, _C_EYE, -1, _AA)


# ── Hand landmark drawing ─────────────────────────────────────────────────────

_HAND_LANDMARK_STYLE = mp_drawing.DrawingSpec(
    color=(80, 230, 80), thickness=1, circle_radius=3
)
_HAND_CONNECTION_STYLE = mp_drawing.DrawingSpec(
    color=(50, 180, 50), thickness=1
)


def draw_hand_landmarks(frame, hand_landmarks):
    """
    Draw hand skeleton onto frame.
    hand_landmarks: mediapipe NormalizedLandmarkList
    """
    if hand_landmarks is None:
        return
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        _HAND_LANDMARK_STYLE,
        _HAND_CONNECTION_STYLE,
    )


# ── Main Overlay class ────────────────────────────────────────────────────────

class Overlay:
    def __init__(self):
        self._audio_ok    = False
        self._alert_sound = None
        self._init_audio()

    def _init_audio(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            sr  = 44100
            t   = np.linspace(0, 0.35, int(sr * 0.35), False)
            wav = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
            self._alert_sound = pygame.sndarray.make_sound(wav)
            self._audio_ok = True
        except Exception as e:
            print(f"[Overlay] Audio init failed: {e}")

    def play_alert(self):
        if self._audio_ok:
            try:
                self._alert_sound.play()
            except Exception:
                pass

    # ── Entry point ───────────────────────────────────────────────────────────

    def draw(self, frame, *, state, signal=None, final_score=None,
             is_alert=False, gesture_label=None, time_remaining=None,
             hand_landmarks=None):
        h, w = frame.shape[:2]

        # Always draw hand skeleton if visible
        draw_hand_landmarks(frame, hand_landmarks)

        if state not in ("ACTIVE", "DEACTIVATE_G1_SEEN"):
            self._draw_inactive(frame, w, h, state, gesture_label, time_remaining)
        else:
            # Draw eye landmarks when active and face is detected
            if signal is not None and hasattr(signal, '_lm_px') and signal._lm_px is not None:
                draw_eye_landmarks(frame, signal._lm_px)
            self._draw_active(frame, w, h, signal, final_score, is_alert,
                              state, time_remaining)

        return frame

    def draw_eyes(self, frame, landmarks_px):
        """Call from main.py to draw eye landmarks separately."""
        if landmarks_px is not None:
            draw_eye_landmarks(frame, landmarks_px)

    # ── Inactive screen ───────────────────────────────────────────────────────

    def _draw_inactive(self, frame, w, h, state, gesture_label, time_remaining):
        # Top status bar
        _panel(frame, 0, 0, w, 48, alpha=0.75, colour=_C_PANEL)
        _txt(frame, "DRIVER FATIGUE SYSTEM", (12, 32), 0.65, _C_DIM)
        _txt(frame, "INACTIVE", (w - 120, 32), 0.65, _C_DIM)

        # Central instruction card
        card_w, card_h = 280, 130
        cx = (w - card_w) // 2
        cy = h // 2 - card_h // 2
        _panel(frame, cx, cy, cx + card_w, cy + card_h, alpha=0.72)

        _txt(frame, "Activate System", (cx + 16, cy + 28), 0.60, _C_WHITE)
        _txt(frame, "1.  Open Palm",   (cx + 16, cy + 58), 0.52, _C_CYAN,  1)
        _txt(frame, "2.  Thumbs Up",   (cx + 16, cy + 82), 0.52, _C_CYAN,  1)
        _txt(frame, "Complete within 5 seconds",
             (cx + 16, cy + 112), 0.38, _C_DIM, 1)

        # Progress indicator when G1 seen
        if state == "GESTURE_1_SEEN" and time_remaining is not None:
            bar_y = cy + card_h + 10
            _panel(frame, cx, bar_y, cx + card_w, bar_y + 28, alpha=0.72)
            fraction = time_remaining / 5.0
            _bar(frame, cx + 6, bar_y + 6, cx + card_w - 6, bar_y + 22,
                 fraction, _C_CALIB)
            _txt(frame, f"Now show Thumbs Up  {time_remaining:.1f}s",
                 (cx + 10, bar_y + 20), 0.42, _C_WHITE, 1)

        # Detected gesture badge
        if gesture_label:
            _panel(frame, 8, h - 40, 200, h - 8, alpha=0.70)
            _txt(frame, f"Gesture: {gesture_label}", (14, h - 16), 0.48, _C_GREEN)

    # ── Active screen ─────────────────────────────────────────────────────────

    def _draw_active(self, frame, w, h, signal, final_score, is_alert,
                     state="ACTIVE", time_remaining=None):
        # ── Top status bar ────────────────────────────────────────────────────
        bar_col = _C_RED if is_alert else _C_GREEN
        _panel(frame, 0, 0, w, 52, alpha=0.82,
               colour=(30, 30, 80) if is_alert else (20, 50, 20))
        cv2.rectangle(frame, (0, 0), (w, 52), bar_col, 2, _AA)

        if signal and signal.calibrating:
            _txt(frame, "CALIBRATING HEAD POSE...", (12, 34), 0.65, _C_CALIB)
        elif is_alert:
            _txt(frame, "  FATIGUE ALERT", (12, 34), 0.72, _C_RED)
        else:
            _txt(frame, "  MONITORING", (12, 34), 0.72, _C_GREEN)

        _txt(frame, "ACTIVE", (w - 90, 34), 0.60, _C_GREEN if not is_alert else _C_RED)

        # Deactivation hint / progress bar
        if state == "DEACTIVATE_G1_SEEN" and time_remaining is not None:
            _panel(frame, w - 230, 56, w - 8, 90, alpha=0.75)
            fraction = time_remaining / 5.0
            _bar(frame, w - 226, 74, w - 12, 86, fraction, _C_CALIB)
            _txt(frame, f"Thumbs Up to stop  {time_remaining:.1f}s",
                 (w - 224, 70), 0.38, _C_WHITE, 1)
        else:
            _txt(frame, "Palm + Thumb to stop", (w - 210, 68), 0.36, _C_DIM, 1)

        if signal is None:
            _panel(frame, 8, 60, 250, 90, alpha=0.65)
            _txt(frame, "No face detected", (14, 82), 0.52, _C_CYAN)
            return

        # ── Alert tags (below status bar) ─────────────────────────────────────
        tag_x = 10
        for alert in signal.alerts:
            tag_label = {"EYE_CLOSURE": "EYE CLOSED",
                         "YAWN":        "YAWN",
                         "HEAD_NOD":    "HEAD NOD",
                         "HEAD_TILT":   "HEAD TILT"}.get(alert, alert)
            tw, _ = cv2.getTextSize(tag_label, _F, 0.42, 1)[0], 0
            tag_w = tw[0] + 16
            _panel(frame, tag_x, 56, tag_x + tag_w, 80,
                   alpha=0.85, colour=(40, 40, 150))
            cv2.rectangle(frame, (tag_x, 56), (tag_x + tag_w, 80),
                          _C_RED, 1, _AA)
            _txt(frame, tag_label, (tag_x + 8, 73), 0.42, _C_WHITE, 1)
            tag_x += tag_w + 6

        # ── Metrics panel (bottom-left) ───────────────────────────────────────
        panel_x, panel_y = 8, h - 125
        _panel(frame, panel_x, panel_y, panel_x + 210, h - 8, alpha=0.70)

        rows = [
            ("EAR",   f"{signal.ear:.3f}",         signal.ear   < 0.25),
            ("MAR",   f"{signal.mar:.3f}",          signal.mar   > 0.60),
            ("Pitch", f"{signal.pitch:+.1f}°",      abs(signal.pitch) > 20),
            ("Roll",  f"{signal.roll:+.1f}°",       abs(signal.roll)  > 20),
        ]
        for i, (label, value, warn) in enumerate(rows):
            y = panel_y + 22 + i * 26
            val_col = _C_RED if warn else _C_WHITE
            _txt(frame, label, (panel_x + 10, y), 0.46, _C_DIM,   1)
            _txt(frame, value, (panel_x + 80, y), 0.46, val_col,  1)
            # Mini indicator dot
            dot_col = _C_RED if warn else _C_GREEN
            cv2.circle(frame, (panel_x + 195, y - 5), 5, dot_col, -1, _AA)

        # ── Fatigue score panel (bottom-right) ────────────────────────────────
        if final_score is not None:
            sx, sy = w - 175, h - 72
            _panel(frame, sx, sy, w - 8, h - 8, alpha=0.70)
            score_col = _C_RED if final_score >= 0.5 else _C_GREEN
            _txt(frame, "Fatigue Score", (sx + 8, sy + 18), 0.42, _C_DIM, 1)
            _txt(frame, f"{final_score * 100:.0f}%",
                 (sx + 8, sy + 46), 0.75, score_col)
            _bar(frame, sx + 8, sy + 52, w - 16, sy + 62,
                 final_score, score_col)
