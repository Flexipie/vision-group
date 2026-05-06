# ── Classical thresholds ──────────────────────────────────────────────────────
EAR_THRESHOLD = 0.25        # below this → eye considered closed
EAR_CONSEC_FRAMES = 20      # frames eye must stay closed to trigger alert

MAR_THRESHOLD = 0.60        # above this → mouth considered open (yawn)
MAR_CONSEC_FRAMES = 15      # frames mouth must stay open to trigger alert

HEAD_PITCH_THRESHOLD = 20.0  # degrees forward/back head nod
HEAD_ROLL_THRESHOLD  = 20.0  # degrees sideways head tilt
HEAD_CONSEC_FRAMES   = 30    # frames head must stay in position to trigger alert
# Euler roll near ±180° is often unstable (mirror camera + solvePnP). Do not latch HEAD_TILT.
HEAD_ROLL_AMBIGUOUS_DEG = 120.0

# In-car side-mount cameras: larger relative roll thresholds reduce false HEAD_TILT.
# Uncomment or adjust vs defaults above if demos from a sharp dash angle stays noisy.

# ── Debug (also set VISION_DEBUG=1) ───────────────────────────────────────────
DEBUG_LOG_EVERY_FRAMES = 15

# ── Gesture activation ────────────────────────────────────────────────────────
GESTURE_SEQUENCE = ["OPEN_PALM", "THUMBS_UP"]
GESTURE_WINDOW_SECONDS = 5   # time window to complete the full sequence

# ── Modern pipeline ───────────────────────────────────────────────────────────
FRAME_BUFFER_SIZE = 30       # frames fed into LSTM
FACE_CROP_SIZE = (112, 112)  # input size for CNN
MODEL_PATH = "models/fatigue_net.pth"

# ── Fusion weights ────────────────────────────────────────────────────────────
# Slightly favor classical vs older 0.4/0.6 so CNN-LSTM false positives dominate less.
FUSION_CLASSICAL_WEIGHT = 0.45
FUSION_MODERN_WEIGHT = 0.55
# Fused-score alert: tune between noisy (≈0.5) vs never firing (≥0.78 when final peaks ~0.75).
FUSION_ALERT_THRESHOLD = 0.72

# ── Classifier (classical ML) ─────────────────────────────────────────────────
CLASSIFIER_PATH = "models/classical_classifier.joblib"

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
