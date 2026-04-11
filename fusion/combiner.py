import config
from classical.detector import FatigueSignal


class FusionCombiner:
    """
    Combines scores from the classical threshold detector and the modern
    CNN-LSTM inferencer into a single fatigue confidence value.

    When the modern pipeline is unavailable (no trained weights), the
    classical score is used with full weight.
    """

    def __init__(self, classical_weight=None, modern_weight=None, threshold=None):
        self.classical_weight = classical_weight or config.FUSION_CLASSICAL_WEIGHT
        self.modern_weight    = modern_weight    or config.FUSION_MODERN_WEIGHT
        self.threshold        = threshold        or config.FUSION_ALERT_THRESHOLD
        self._log             = []   # stores per-frame scores for report export

    def combine(self, classical_signal: FatigueSignal, modern_score):
        """
        classical_signal : FatigueSignal from ThresholdDetector
        modern_score     : float in [0,1] from ModernInferencer, or None

        Returns (final_score, is_alert) where:
          final_score : float in [0, 1]
          is_alert    : bool
        """
        classical_score = classical_signal.classical_score

        if modern_score is not None:
            w_c = self.classical_weight
            w_m = self.modern_weight
            final = (w_c * classical_score + w_m * modern_score) / (w_c + w_m)
        else:
            final = classical_score

        is_alert = (final >= self.threshold) or bool(classical_signal.alerts)

        self._log.append({
            "classical": classical_score,
            "modern":    modern_score,
            "final":     final,
            "alerts":    list(classical_signal.alerts),
        })

        return float(final), is_alert

    def export_log(self, path="results/fusion_log.csv"):
        """Save per-frame score log to CSV for the report."""
        import os, csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["classical", "modern", "final", "alerts"])
            writer.writeheader()
            writer.writerows(self._log)
        print(f"[FusionCombiner] Log saved to {path}")
