import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "vision_group_matplotlib"),
)

import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


class GestureDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

    def process(self, bgr_frame):
        """
        Returns (gesture_label, hand_landmarks) where:
          gesture_label : "OPEN_PALM" | "THUMBS_UP" | None
          hand_landmarks: mediapipe NormalizedLandmarkList or None (for drawing)
        """
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None, None

        raw_lm = results.multi_hand_landmarks[0]
        pts    = np.array([[l.x, l.y, l.z] for l in raw_lm.landmark])

        return self._classify(pts), raw_lm

    def _finger_extended(self, pts, tip_idx, pip_idx):
        """True if fingertip is above (lower y) its PIP joint in image coords."""
        return pts[tip_idx][1] < pts[pip_idx][1]

    def _classify(self, pts):
        index_up   = self._finger_extended(pts, INDEX_TIP,  INDEX_PIP)
        middle_up  = self._finger_extended(pts, MIDDLE_TIP, MIDDLE_PIP)
        ring_up    = self._finger_extended(pts, RING_TIP,   RING_PIP)
        pinky_up   = self._finger_extended(pts, PINKY_TIP,  PINKY_PIP)

        # Thumb: extended when tip is to the left of IP joint (right hand)
        # Use x-axis distance; works for both orientations roughly
        thumb_up_gesture = (
            pts[THUMB_TIP][1] < pts[THUMB_IP][1]   # thumb tip above thumb IP
            and not index_up
            and not middle_up
            and not ring_up
            and not pinky_up
        )

        open_palm = index_up and middle_up and ring_up and pinky_up

        if thumb_up_gesture:
            return "THUMBS_UP"
        if open_palm:
            return "OPEN_PALM"
        return None

    def close(self):
        self.hands.close()
