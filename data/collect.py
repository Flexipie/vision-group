"""
Data collection script — run this inside a parked car.

Usage:
    python data/collect.py --label alert   --clips 10 --duration 30
    python data/collect.py --label drowsy  --clips 10 --duration 30
    python data/collect.py --label yawn    --clips 5  --duration 20
    python data/collect.py --label head_nod --clips 5 --duration 20

Controls during recording:
    SPACE  → start / stop a clip
    Q      → quit
"""

import argparse
import os
import time
import cv2
import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label",    required=True,
                   choices=["alert", "drowsy", "yawn", "head_nod"],
                   help="Fatigue category being recorded")
    p.add_argument("--clips",    type=int, default=10, help="Number of clips to record")
    p.add_argument("--duration", type=int, default=30, help="Max seconds per clip")
    p.add_argument("--out",      default="data/raw",   help="Output directory")
    p.add_argument("--camera",   type=int, default=config.CAMERA_INDEX)
    return p.parse_args()


def main():
    args  = parse_args()
    out   = os.path.join(args.out, args.label)
    os.makedirs(out, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps    = 30
    clip_n = 0
    writer = None
    recording = False
    t_start   = 0

    print(f"\n[collect] Label: {args.label}")
    print(f"[collect] Saving clips to: {out}")
    print("[collect] Press SPACE to start/stop recording. Q to quit.\n")

    while clip_n < args.clips:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        elapsed = time.time() - t_start if recording else 0

        # Auto-stop if max duration reached
        if recording and elapsed >= args.duration:
            writer.release()
            writer = None
            recording = False
            clip_n += 1
            print(f"[collect] Clip {clip_n}/{args.clips} saved (auto-stop).")

        # UI
        status = f"RECORDING {elapsed:.1f}s / {args.duration}s" if recording else "STANDBY — press SPACE"
        colour = (0, 0, 220) if recording else (0, 200, 0)
        cv2.putText(display, f"[{args.label.upper()}]  {status}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.putText(display, f"Clips: {clip_n}/{args.clips}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if recording and writer:
            writer.write(frame)

        cv2.imshow("Data Collection", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if not recording:
                fname  = os.path.join(out, f"{args.label}_{clip_n+1:03d}.mp4")
                writer = cv2.VideoWriter(fname, fourcc, fps,
                                         (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                recording = True
                t_start   = time.time()
                print(f"[collect] Recording clip {clip_n+1} → {fname}")
            else:
                writer.release()
                writer    = None
                recording = False
                clip_n   += 1
                print(f"[collect] Clip {clip_n}/{args.clips} saved.")
        elif key == ord("q"):
            break

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[collect] Done. {clip_n} clips saved to '{out}'.")


if __name__ == "__main__":
    main()
