import numpy as np
import cv2
from classical.landmarks import LEFT_EYE, RIGHT_EYE, MOUTH_OUTER, HEAD_POSE_POINTS


def _refine_pose_rvec(object_points_n3, image_points_n2, camera_matrix, dist_coeffs, rvec, tvec):
    """Refine PnP with LM when available."""
    refined = getattr(cv2, "solvePnPRefineLM", None)
    if refined is None:
        return rvec, tvec
    try:
        rvec_ref, tvec_ref = refined(
            object_points_n3,
            image_points_n2.astype(np.float64),
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
        )
        return rvec_ref, tvec_ref
    except cv2.error:
        return rvec, tvec


# ── 3-D reference face model (generic, in mm) ─────────────────────────────────
# Ordered to match HEAD_POSE_POINTS: nose tip, chin, left eye corner,
# right eye corner, left mouth, right mouth.
# Uses OpenCV image convention: Y axis points DOWN, so:
#   chin  is below the nose → positive Y
#   eyes  are above the nose → negative Y
#   mouth is below the nose → positive Y
_3D_FACE_MODEL = np.array([
    [ 0.0,    0.0,    0.0  ],  # Nose tip
    [ 0.0,   63.6,  -12.5 ],  # Chin      (below  → +Y)
    [-43.3, -32.7,  -26.0 ],  # Left eye  (above  → -Y)
    [ 43.3, -32.7,  -26.0 ],  # Right eye (above  → -Y)
    [-28.9,  28.9,  -24.1 ],  # Left mouth corner  (+Y)
    [ 28.9,  28.9,  -24.1 ],  # Right mouth corner (+Y)
], dtype=np.float64)


def _dist(a, b):
    return np.linalg.norm(a - b)


def compute_ear(landmarks_px):
    """
    Eye Aspect Ratio averaged over both eyes.
    landmarks_px: (468, 2) float32 pixel coords
    Returns float EAR in [0, ~0.4]; lower = more closed.
    """
    def _eye_ear(indices):
        p = landmarks_px[indices]
        # Vertical distances (pairs: 1-5, 2-4)
        v1 = _dist(p[1], p[5])
        v2 = _dist(p[2], p[4])
        # Horizontal distance (0-3)
        h  = _dist(p[0], p[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    left  = _eye_ear(LEFT_EYE)
    right = _eye_ear(RIGHT_EYE)
    return float((left + right) / 2.0)


def compute_mar(landmarks_px):
    """
    Mouth Aspect Ratio.
    landmarks_px: (468, 2) float32 pixel coords
    Returns float MAR; higher = more open.
    """
    m = MOUTH_OUTER
    # Three vertical distances
    v1 = _dist(landmarks_px[m["top_left"][0]],   landmarks_px[m["bottom_left"][0]])
    v2 = _dist(landmarks_px[m["top_inner"][0]],  landmarks_px[m["bottom_inner"][0]])
    v3 = _dist(landmarks_px[m["top_right"][0]],  landmarks_px[m["bottom_right"][0]])
    # Horizontal distance
    h  = _dist(landmarks_px[m["left_corner"][0]], landmarks_px[m["right_corner"][0]])
    return float((v1 + v2 + v3) / (2.0 * h + 1e-6))


def _wrap180(angle):
    """Normalise any angle to the (-180, 180] range."""
    return ((float(angle) + 180.0) % 360.0) - 180.0


def compute_head_pose(landmarks_px, frame_shape):
    """
    Estimates head pose using solvePnP.

    Returns (pitch, yaw, roll) in degrees, each normalised to (-180, 180].
      pitch > 0  → head nodding down
      yaw   > 0  → head turning right
      roll  > 0  → head tilting to right shoulder
    Returns (0.0, 0.0, 0.0) on solver failure.
    """
    h, w = frame_shape
    image_points = landmarks_px[HEAD_POSE_POINTS].astype(np.float64)

    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0,        w / 2.0],
        [0,        focal_length, h / 2.0],
        [0,        0,            1.0    ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        _3D_FACE_MODEL,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rvec, tvec = _refine_pose_rvec(
        _3D_FACE_MODEL, image_points, camera_matrix, dist_coeffs, rvec, tvec
    )

    rmat, _ = cv2.Rodrigues(rvec)

    # XYZ (Tait-Bryan) decomposition: Rx * Ry * Rz
    # This convention is stable for typical head-pose ranges.
    pitch = _wrap180(np.degrees(np.arctan2( rmat[2, 1], rmat[2, 2])))
    sy    = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    yaw   = _wrap180(np.degrees(np.arctan2(-rmat[2, 0], sy)))
    roll  = _wrap180(np.degrees(np.arctan2( rmat[1, 0], rmat[0, 0])))

    return float(pitch), float(yaw), float(roll)


def extract_feature_vector(landmarks_px, frame_shape, blink_rate=0.0):
    """
    Returns a 6-element feature vector for the classical ML classifier.
    [EAR, MAR, pitch, yaw, roll, blink_rate]
    blink_rate: blinks per second (computed externally from frame counter).
    """
    ear = compute_ear(landmarks_px)
    mar = compute_mar(landmarks_px)
    pitch, yaw, roll = compute_head_pose(landmarks_px, frame_shape)
    return np.array([ear, mar, pitch, yaw, roll, blink_rate], dtype=np.float32)
