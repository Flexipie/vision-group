import cv2
import sys
import config
from gesture.detector import GestureDetector
from gesture.state_machine import GestureStateMachine
from classical.landmarks import FaceLandmarkExtractor
from classical.detector import ThresholdDetector
from classical.classifier import ClassicalClassifier
from classical.features import extract_feature_vector
from modern.infer import ModernInferencer
from fusion.combiner import FusionCombiner
from ui.overlay import Overlay


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    gesture_detector  = GestureDetector()
    state_machine     = GestureStateMachine()
    face_extractor    = FaceLandmarkExtractor()
    threshold_det     = ThresholdDetector()
    classifier        = ClassicalClassifier()   # loads saved model if present
    modern_inf        = ModernInferencer()
    fusion            = FusionCombiner()
    overlay           = Overlay()

    alert_playing     = False
    prev_state        = state_machine.state

    print("[INFO] System started. Perform OPEN_PALM → THUMBS_UP to activate.")
    print("[INFO] Press 'q' to quit, 'r' to reset activation.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)   # mirror for natural interaction

        # ── 1. Gesture activation ─────────────────────────────────────────────
        gesture_label, hand_landmarks = gesture_detector.process(frame)
        state_machine.update(gesture_label)
        state = state_machine.state

        # Detect gesture-driven deactivation
        if prev_state != "INACTIVE" and state == "INACTIVE":
            threshold_det.reset()
            alert_playing = False
            print("[INFO] System deactivated via gesture.")
        prev_state = state

        signal       = None
        final_score  = None
        is_alert     = False
        lm_px        = None

        # ── 2. Fatigue detection (only when active) ───────────────────────────
        if state_machine.is_active():
            lm_px, lm_norm, frame_shape = face_extractor.process(frame)

            if lm_px is not None:
                # Classical threshold detection
                signal = threshold_det.detect(lm_px, frame_shape)

                # Classical ML prediction (if model available)
                fv = extract_feature_vector(lm_px, frame_shape, signal.blink_rate)
                ml_score = classifier.predict(fv)
                if ml_score is not None:
                    # Blend threshold score with ML score
                    signal.classical_score = 0.5 * signal.classical_score + 0.5 * float(ml_score)

                # Modern CNN-LSTM prediction
                x1, y1, x2, y2 = face_extractor.get_face_bbox(lm_px)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_crop  = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
                modern_score = modern_inf.predict(face_crop) if face_crop is not None else None

                # Fuse scores
                final_score, is_alert = fusion.combine(signal, modern_score)

                # Audio alert (non-repeating until cleared)
                if is_alert and not alert_playing:
                    overlay.play_alert()
                    alert_playing = True
                elif not is_alert:
                    alert_playing = False

        # ── 3. Draw landmarks + overlay ───────────────────────────────────────
        overlay.draw_eyes(frame, lm_px)
        overlay.draw(
            frame,
            state=state,
            signal=signal,
            final_score=final_score,
            is_alert=is_alert,
            gesture_label=gesture_label,
            time_remaining=state_machine.time_remaining(),
            hand_landmarks=hand_landmarks,
        )

        cv2.imshow("Driver Fatigue Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            state_machine.reset()
            threshold_det.reset()
            alert_playing = False
            print("[INFO] Reset — system deactivated.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    gesture_detector.close()
    face_extractor.close()
    fusion.export_log()


if __name__ == "__main__":
    main()
