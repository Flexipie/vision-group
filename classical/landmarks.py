import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# ── Landmark index groups ─────────────────────────────────────────────────────
# MediaPipe Face Mesh 468-point model

# Left eye (from viewer's perspective)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
# Right eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Outer lip landmarks for MAR
# Vertical pairs: top/bottom at three positions
# Horizontal: left corner / right corner
MOUTH_OUTER = {
    "top_inner":    [13],
    "bottom_inner": [14],
    "top_left":     [82],
    "bottom_left":  [87],
    "top_right":    [312],
    "bottom_right": [317],
    "left_corner":  [61],
    "right_corner": [291],
}

# 6 points used for head pose (indices into the 468 face mesh)
HEAD_POSE_POINTS = [
    1,    # Nose tip
    152,  # Chin
    226,  # Left eye left corner
    446,  # Right eye right corner
    57,   # Left mouth corner
    287,  # Right mouth corner
]


class FaceLandmarkExtractor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, bgr_frame):
        """
        Returns (landmarks_px, landmarks_norm, frame_shape) or (None, None, shape).

        landmarks_px:   (468, 2) array of (x, y) in pixel coords
        landmarks_norm: (468, 3) array of (x, y, z) normalised [0,1]
        """
        import cv2
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None, (h, w)

        lm = results.multi_face_landmarks[0].landmark
        norm = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
        px   = (norm[:, :2] * np.array([w, h])).astype(np.float32)
        return px, norm, (h, w)

    def get_face_bbox(self, landmarks_px, padding=20):
        """Returns (x1, y1, x2, y2) bounding box around face landmarks."""
        x1 = int(landmarks_px[:, 0].min()) - padding
        y1 = int(landmarks_px[:, 1].min()) - padding
        x2 = int(landmarks_px[:, 0].max()) + padding
        y2 = int(landmarks_px[:, 1].max()) + padding
        return x1, y1, x2, y2

    def close(self):
        self.face_mesh.close()
