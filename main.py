import argparse
import os
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


def parse_args():
    parser = argparse.ArgumentParser(description="Driver fatigue detection (in-car demo).")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="On-screen debug panel (threshold / ML / CNN-LSTM / alert drivers).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    debug = bool(args.debug) or os.environ.get("VISION_DEBUG", "").strip() == "1"

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
    if debug:
        print("[INFO] Debug overlay enabled (VISION_DEBUG=1 or --debug).")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)   # mirror for natural interaction

        debug_info = None

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
                signal = threshold_det.detect(lm_px, frame_shape)
                threshold_score = signal.classical_score

                fv = extract_feature_vector(lm_px, frame_shape, signal.blink_rate)
                ml_score = classifier.predict(fv)
                if ml_score is not None:
                    signal.classical_score = (
                        0.5 * threshold_score + 0.5 * float(ml_score)
                    )
                classical_fused = signal.classical_score

                x1, y1, x2, y2 = face_extractor.get_face_bbox(lm_px)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_crop = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
                modern_score = (
                    modern_inf.predict(face_crop)
                    if face_crop is not None
                    else None
                )

                final_score, is_alert = fusion.combine(signal, modern_score)

                rule_alert = bool(signal.alerts)
                score_alert = (
                    final_score is not None
                    and final_score >= config.FUSION_ALERT_THRESHOLD
                )
                if rule_alert and score_alert:
                    alert_summary = "RULE+SCORE"
                elif rule_alert and signal.alerts:
                    alert_summary = "RULE:" + signal.alerts[0]
                elif score_alert:
                    alert_summary = f"SCORE>={config.FUSION_ALERT_THRESHOLD:.2f}"
                else:
                    alert_summary = "---"

                if debug:
                    debug_info = {
                        "threshold_score": threshold_score,
                        "ml_score": ml_score,
                        "classical_fused": classical_fused,
                        "modern_score": modern_score,
                        "modern_available": modern_inf.available,
                        "modern_buf": modern_inf.buffer_len(),
                        "modern_cap": modern_inf.buffer_capacity(),
                        "final_score": final_score,
                        "alert_summary": alert_summary,
                        "yaw": signal.yaw,
                    }

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
            debug_info=debug_info if debug else None,
        )

        if (
            debug
            and debug_info is not None
            and frame_idx % config.DEBUG_LOG_EVERY_FRAMES == 0
        ):
            ml = debug_info["ml_score"]
            ms = debug_info["modern_score"]
            ml_s = f"{ml:.2f}" if ml is not None else "--"
            mod_s = f"{ms:.2f}" if ms is not None else "--"
            print(
                f"[debug] Thr={debug_info['threshold_score']:.2f} "
                f"ML={ml_s} Cls={debug_info['classical_fused']:.2f} "
                f"Mod={mod_s} buf={debug_info['modern_buf']}/{debug_info['modern_cap']} "
                f"final={debug_info['final_score']:.2f} "
                f"yaw={debug_info['yaw']:+.1f} {debug_info['alert_summary']}"
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

        frame_idx += 1

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    gesture_detector.close()
    face_extractor.close()
    fusion.export_log()


if __name__ == "__main__":
    main()
