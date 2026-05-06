"""
Train and evaluate the classical ML classifier from collected in-car frames.

Example:
    python -m classical.train --data-root frames/fatigue --refresh-features
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import config
from classical.classifier import ClassicalClassifier
from classical.features import extract_feature_vector
from classical.landmarks import FaceLandmarkExtractor
from data.frame_dataset import (
    binary_counts,
    class_counts,
    limit_per_label,
    list_fatigue_frames,
)


FEATURES_PATH = Path("results/classical_features.csv")
METRICS_PATH = Path("results/classical_metrics.json")
CONFUSION_PATH = Path("results/classical_confusion_matrix.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="frames/fatigue")
    parser.add_argument("--features-out", default=str(FEATURES_PATH))
    parser.add_argument("--metrics-out", default=str(METRICS_PATH))
    parser.add_argument("--confusion-out", default=str(CONFUSION_PATH))
    parser.add_argument("--model-out", default=config.CLASSIFIER_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-per-label", type=int, default=None,
                        help="Optional speed limit for quick experiments.")
    parser.add_argument("--refresh-features", action="store_true",
                        help="Recompute MediaPipe features instead of reusing the cached CSV.")
    return parser.parse_args()


def _write_feature_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _read_feature_csv(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_features(data_root, features_out, max_per_label=None):
    records = limit_per_label(list_fatigue_frames(data_root), max_per_label)
    if not records:
        raise RuntimeError(f"No fatigue frames found under '{data_root}'.")

    print(f"[classical.train] Per-folder counts: {class_counts(records)}")
    print(f"[classical.train] Binary counts: {binary_counts(records)}")

    extractor = FaceLandmarkExtractor()
    rows = []
    skipped = 0

    try:
        for idx, record in enumerate(records, start=1):
            bgr = cv2.imread(str(record.path))
            if bgr is None:
                skipped += 1
                continue

            lm_px, _, shape = extractor.process(bgr)
            if lm_px is None:
                skipped += 1
                continue

            fv = extract_feature_vector(lm_px, shape)
            rows.append({
                "path": str(record.path),
                "label": int(record.label),
                "label_name": record.label_name,
                "clip_id": record.clip_id,
                "EAR": float(fv[0]),
                "MAR": float(fv[1]),
                "pitch": float(fv[2]),
                "yaw": float(fv[3]),
                "roll": float(fv[4]),
                "blink_rate": float(fv[5]),
            })

            if idx % 250 == 0:
                print(f"[classical.train] Processed {idx}/{len(records)} frames...")
    finally:
        extractor.close()

    if not rows:
        raise RuntimeError("No usable face-landmark features were extracted.")

    _write_feature_csv(rows, features_out)
    print(f"[classical.train] Saved {len(rows)} feature rows to {features_out}")
    if skipped:
        print(f"[classical.train] Skipped {skipped} frames without usable input.")
    return rows


def rows_to_arrays(rows):
    feature_names = ["EAR", "MAR", "pitch", "yaw", "roll", "blink_rate"]
    x = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    return x, y


def train_and_evaluate(rows, args):
    x, y = rows_to_arrays(rows)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Need both alert and drowsy samples to train the classifier.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    clf = ClassicalClassifier()
    clf.train(x_train, y_train)
    clf.save(args.model_out)

    probabilities = clf.predict(x_test)
    y_score = np.asarray(probabilities, dtype=np.float32)
    y_pred = (y_score >= 0.5).astype(np.int64)

    metrics = {
        "model_path": args.model_out,
        "features_path": args.features_out,
        "n_samples": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_size": float(args.test_size),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_drowsy": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_drowsy": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_drowsy": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["alert", "drowsy"],
            zero_division=0,
            output_dict=True,
        ),
    }

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    confusion_path = Path(args.confusion_out)
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    with confusion_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", "alert", "drowsy"])
        writer.writerow(["alert", int(cm[0, 0]), int(cm[0, 1])])
        writer.writerow(["drowsy", int(cm[1, 0]), int(cm[1, 1])])

    print(f"[classical.train] Saved metrics to {metrics_path}")
    print(f"[classical.train] Saved confusion matrix to {confusion_path}")
    print(
        "[classical.train] "
        f"accuracy={metrics['accuracy']:.3f}, "
        f"f1_drowsy={metrics['f1_drowsy']:.3f}, "
        f"roc_auc={metrics['roc_auc']:.3f}"
    )


def main():
    args = parse_args()
    features_path = Path(args.features_out)

    if features_path.exists() and not args.refresh_features:
        print(f"[classical.train] Reusing cached features from {features_path}")
        rows = _read_feature_csv(features_path)
    else:
        rows = extract_features(args.data_root, features_path, args.max_per_label)

    train_and_evaluate(rows, args)


if __name__ == "__main__":
    main()
