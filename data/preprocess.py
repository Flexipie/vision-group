"""
Preprocessing script — run after data/collect.py.

1. Extracts frames from raw video clips.
2. Detects and crops face regions using MediaPipe Face Mesh.
3. Saves cropped frames to data/frames/{label}/ for CNN-LSTM training.
4. Computes and saves feature CSVs (EAR, MAR, pose) for classical classifier.

Usage:
    python data/preprocess.py --fps 5
"""

import argparse
import os
import csv
import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from classical.landmarks import FaceLandmarkExtractor
from classical.features import extract_feature_vector

RAW_ROOT    = "data/raw"
FRAMES_ROOT = "data/frames"
CSV_PATH    = "data/features.csv"

LABELS = ["alert", "drowsy", "yawn", "head_nod"]

# Map multi-class labels to binary (0=alert, 1=drowsy)
LABEL_MAP = {
    "alert":    0,
    "drowsy":   1,
    "yawn":     1,
    "head_nod": 1,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=int, default=5,
                   help="Frames per second to extract from video")
    return p.parse_args()


def extract_frames(video_path, target_fps):
    """Yield (frame_index, bgr_frame) at target_fps from a video."""
    cap  = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step    = max(1, int(src_fps / target_fps))
    idx     = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield idx, frame
        idx += 1
    cap.release()


def main():
    args     = parse_args()
    extractor = FaceLandmarkExtractor()
    csv_rows  = []

    for label in LABELS:
        raw_dir    = os.path.join(RAW_ROOT, label)
        frames_dir = os.path.join(FRAMES_ROOT, label)

        if not os.path.isdir(raw_dir):
            print(f"[preprocess] Skipping '{label}' — no directory at {raw_dir}")
            continue

        os.makedirs(frames_dir, exist_ok=True)
        video_files = [f for f in os.listdir(raw_dir) if f.endswith((".mp4", ".avi", ".mov"))]
        print(f"[preprocess] {label}: {len(video_files)} videos")

        frame_global = 0
        for vf in sorted(video_files):
            vpath = os.path.join(raw_dir, vf)
            for _, bgr in extract_frames(vpath, args.fps):
                lm_px, _, shape = extractor.process(bgr)
                if lm_px is None:
                    continue

                # ── Save face crop ────────────────────────────────────────────
                x1, y1, x2, y2 = extractor.get_face_bbox(lm_px)
                h, w = bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop   = bgr[y1:y2, x1:x2]
                    crop   = cv2.resize(crop, config.FACE_CROP_SIZE)
                    fname  = f"{label}_{frame_global:06d}.jpg"
                    fpath  = os.path.join(frames_dir, fname)
                    cv2.imwrite(fpath, crop)

                # ── Compute feature vector ────────────────────────────────────
                fv = extract_feature_vector(lm_px, shape)
                csv_rows.append({
                    "label":      LABEL_MAP[label],
                    "label_name": label,
                    "EAR":        fv[0],
                    "MAR":        fv[1],
                    "pitch":      fv[2],
                    "yaw":        fv[3],
                    "roll":       fv[4],
                    "blink_rate": fv[5],
                })

                frame_global += 1

        print(f"[preprocess]   → {frame_global} frames extracted")

    # ── Save feature CSV ──────────────────────────────────────────────────────
    if csv_rows:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[preprocess] Feature CSV saved to '{CSV_PATH}' ({len(csv_rows)} rows)")

    extractor.close()
    print("[preprocess] Done.")


if __name__ == "__main__":
    main()
