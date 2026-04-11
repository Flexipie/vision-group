# =============================================================================
# PLACEHOLDER — requires own labeled dataset to train.
#
# How to use once data is collected:
#   1. Run data/collect.py to record video clips in the car.
#   2. Run data/preprocess.py to extract frames and compute feature CSVs.
#   3. Call ClassicalClassifier().train(features, labels) and .save().
#   4. The ThresholdDetector in detector.py will automatically incorporate
#      predictions from this classifier via the FusionCombiner.
# =============================================================================

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import config


class ClassicalClassifier:
    """
    SVM + Random Forest ensemble classifier trained on EAR/MAR/pose features.
    Falls back to None predictions when no saved model exists.

    Feature vector: [EAR, MAR, pitch, yaw, roll, blink_rate]
    Labels: 0 = alert, 1 = drowsy
    """

    def __init__(self):
        self._pipeline = None
        self.available = False
        self._try_load(config.CLASSIFIER_PATH)

    def _build_pipeline(self):
        svm = SVC(kernel="rbf", probability=True, C=10, gamma="scale")
        rf  = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        ensemble = VotingClassifier(
            estimators=[("svm", svm), ("rf", rf)],
            voting="soft",
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    ensemble),
        ])

    def train(self, features, labels):
        """
        features: (N, 6) array — [EAR, MAR, pitch, yaw, roll, blink_rate]
        labels:   (N,)   array — 0=alert, 1=drowsy
        """
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(features, labels)
        self.available = True
        print(f"[ClassicalClassifier] Trained on {len(labels)} samples.")

    def predict(self, feature_vector):
        """
        feature_vector: (6,) array or (N, 6) array
        Returns drowsy probability in [0, 1], or None if not trained.
        """
        if not self.available:
            return None
        fv = np.atleast_2d(feature_vector)
        prob = self._pipeline.predict_proba(fv)[:, 1]
        return float(prob[0]) if fv.shape[0] == 1 else prob

    def save(self, path=None):
        path = path or config.CLASSIFIER_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self._pipeline, path)
        print(f"[ClassicalClassifier] Saved to {path}")

    def _try_load(self, path):
        if os.path.exists(path):
            try:
                self._pipeline = joblib.load(path)
                self.available = True
                print(f"[ClassicalClassifier] Loaded from {path}")
            except Exception as e:
                print(f"[ClassicalClassifier] Failed to load {path}: {e}")
