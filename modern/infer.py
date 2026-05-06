import os
import collections
import numpy as np
import torch
import cv2
import config
from modern.model import FatigueNet

# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_crop(bgr_crop):
    """Resize, normalise, and convert face crop to CHW float32 tensor."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, config.FACE_CROP_SIZE).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))  # (3, H, W)


class ModernInferencer:
    """
    Real-time CNN-LSTM fatigue inference.

    If no weights file is found at config.MODEL_PATH, self.available = False
    and predict() always returns None — the system falls back to classical only.
    """

    def __init__(self, device=None):
        self.available = False
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._buffer   = collections.deque(maxlen=config.FRAME_BUFFER_SIZE)
        self._model    = None
        self._try_load()

    def _try_load(self):
        path = config.MODEL_PATH
        if not os.path.exists(path):
            print(f"[ModernInferencer] No weights at '{path}' — running without modern pipeline.")
            return
        try:
            self._model = FatigueNet()
            state = torch.load(path, map_location=self.device)
            self._model.load_state_dict(state)
            self._model.to(self.device)
            self._model.eval()
            self.available = True
            print(f"[ModernInferencer] Loaded weights from '{path}'.")
        except Exception as e:
            print(f"[ModernInferencer] Failed to load weights: {e}")

    def buffer_len(self):
        """Frames currently in the temporal buffer."""
        return len(self._buffer)

    def buffer_capacity(self):
        """Required frames before CNN-LSTM returns a score."""
        return config.FRAME_BUFFER_SIZE

    def predict(self, bgr_face_crop):
        """
        Feed one face crop (BGR numpy array) per frame.
        Returns drowsy probability float in [0, 1], or None if not available
        or buffer not yet full.
        """
        if not self.available:
            return None

        tensor = _preprocess_crop(bgr_face_crop)
        self._buffer.append(tensor)

        if len(self._buffer) < config.FRAME_BUFFER_SIZE:
            return None  # wait until buffer is full

        # Stack buffer into (1, T, C, H, W)
        seq = torch.stack(list(self._buffer)).unsqueeze(0).to(self.device)
        prob = self._model.predict_proba(seq)
        return float(prob[0].item())
