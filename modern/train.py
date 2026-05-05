"""
Train and evaluate the CNN-LSTM fatigue model from collected in-car frames.

Example quick run:
    python -m modern.train --data-root frames/fatigue --epochs 3 --stride 10 --freeze-cnn

Example fuller run:
    python -m modern.train --data-root frames/fatigue --epochs 15 --stride 3 --pretrained --freeze-cnn --precompute-features
"""

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import config
from data.frame_dataset import FrameRecord, binary_counts, class_counts, list_fatigue_frames
from modern.model import FatigueNet


DATA_ROOT = "frames/fatigue"
HISTORY_PATH = Path("results/modern_training_history.csv")
METRICS_PATH = Path("results/modern_metrics.json")
CONFUSION_PATH = Path("results/modern_confusion_matrix.csv")
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=config.FRAME_BUFFER_SIZE)
    parser.add_argument("--stride", type=int, default=5,
                        help="Sliding-window stride in frames.")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-sequences-per-class", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet-pretrained MobileNetV2 weights if available.")
    parser.add_argument("--freeze-cnn", action="store_true",
                        help="Train only the LSTM/classifier head for faster coursework runs.")
    parser.add_argument("--precompute-features", action="store_true",
                        help="Cache CNN features first when --freeze-cnn is used.")
    parser.add_argument("--model-out", default=config.MODEL_PATH)
    parser.add_argument("--history-out", default=str(HISTORY_PATH))
    parser.add_argument("--metrics-out", default=str(METRICS_PATH))
    parser.add_argument("--confusion-out", default=str(CONFUSION_PATH))
    return parser.parse_args()


class FatigueSequenceDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label, _clip_id = self.samples[idx]
        frames = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            frames.append(self.transform(img))
        return torch.stack(frames), int(label)


class PrecomputedFeatureDataset(Dataset):
    def __init__(self, samples, feature_cache):
        self.samples = samples
        self.feature_cache = feature_cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label, _clip_id = self.samples[idx]
        feats = [self.feature_cache[str(path)] for path in paths]
        return torch.stack(feats), int(label)


def _build_transforms(train: bool):
    steps = [
        transforms.Resize(config.FACE_CROP_SIZE),
    ]
    if train:
        steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.RandomRotation(8),
        ])
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(steps)


def _make_sequences(records: Sequence[FrameRecord], seq_len: int, stride: int):
    by_clip: Dict[str, List[FrameRecord]] = {}
    for record in records:
        by_clip.setdefault(record.clip_id, []).append(record)

    samples = []
    for clip_id, clip_records in by_clip.items():
        clip_records = sorted(clip_records, key=lambda r: r.path.name)
        if len(clip_records) < seq_len:
            continue
        label = clip_records[0].label
        for start in range(0, len(clip_records) - seq_len + 1, stride):
            paths = [record.path for record in clip_records[start:start + seq_len]]
            samples.append((paths, label, clip_id))

    return samples


def _split_by_clip(samples, val_split: float, seed: int = SEED):
    rng = random.Random(seed)
    clips_by_label: Dict[int, List[str]] = {0: [], 1: []}
    labels_by_clip = {}

    for _paths, label, clip_id in samples:
        labels_by_clip[clip_id] = label

    for clip_id, label in labels_by_clip.items():
        clips_by_label[label].append(clip_id)

    val_clips = set()
    for label, clips in clips_by_label.items():
        rng.shuffle(clips)
        n_val = max(1, int(round(len(clips) * val_split))) if len(clips) > 1 else 0
        val_clips.update(clips[:n_val])

    train_samples = [sample for sample in samples if sample[2] not in val_clips]
    val_samples = [sample for sample in samples if sample[2] in val_clips]
    return train_samples, val_samples


def _limit_sequences(samples, max_per_class):
    if max_per_class is None or max_per_class <= 0:
        return samples
    limited = []
    counts = {0: 0, 1: 0}
    for sample in samples:
        label = sample[1]
        if counts[label] < max_per_class:
            limited.append(sample)
            counts[label] += 1
    return limited


def _write_history(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", "alert", "drowsy"])
        writer.writerow(["alert", int(cm[0, 0]), int(cm[0, 1])])
        writer.writerow(["drowsy", int(cm[1, 0]), int(cm[1, 1])])


def _precompute_cnn_features(model, samples, transform, device):
    unique_paths = sorted({path for paths, _label, _clip_id in samples for path in paths})
    cache = {}
    model.eval()

    with torch.no_grad():
        for idx, path in enumerate(unique_paths, start=1):
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            feat = model.cnn(x)
            feat = model.pool(feat).view(-1).cpu()
            cache[str(path)] = feat
            if idx % 250 == 0:
                print(f"[modern.train] Precomputed CNN features {idx}/{len(unique_paths)}...")

    print(f"[modern.train] Precomputed CNN features for {len(unique_paths)} unique frames.")
    return cache


def evaluate(model, loader, criterion, device, precomputed=False):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for seqs, labels in loader:
            seqs = seqs.to(device)
            labels = labels.to(device)
            logits = model.forward_from_features(seqs) if precomputed else model(seqs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            pred = logits.argmax(1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    total = max(1, len(y_true))
    return {
        "loss": total_loss / total,
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "precision_drowsy": precision_score(y_true, y_pred, zero_division=0) if y_true else 0.0,
        "recall_drowsy": recall_score(y_true, y_pred, zero_division=0) if y_true else 0.0,
        "f1_drowsy": f1_score(y_true, y_pred, zero_division=0) if y_true else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    records = list_fatigue_frames(args.data_root)
    if not records:
        raise RuntimeError(f"No fatigue frames found under '{args.data_root}'.")

    print(f"[modern.train] Per-folder counts: {class_counts(records)}")
    print(f"[modern.train] Binary counts: {binary_counts(records)}")

    samples = _make_sequences(records, args.seq_len, args.stride)
    samples = _limit_sequences(samples, args.max_sequences_per_class)
    if not samples:
        raise RuntimeError(
            f"No frame sequences could be built. Need at least {args.seq_len} frames per clip."
        )

    train_samples, val_samples = _split_by_clip(samples, args.val_split)
    if not train_samples or not val_samples:
        raise RuntimeError("Train/validation split is empty; lower --val-split or collect more clips.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[modern.train] Using device: {device}")

    model = FatigueNet(pretrained=args.pretrained).to(device)
    if args.freeze_cnn:
        for param in model.cnn.parameters():
            param.requires_grad = False
        print("[modern.train] CNN backbone frozen.")

    use_precomputed = bool(args.freeze_cnn and args.precompute_features)
    print(f"[modern.train] Train sequences: {len(train_samples)}")
    print(f"[modern.train] Val sequences:   {len(val_samples)}")

    if use_precomputed:
        eval_tf = _build_transforms(train=False)
        feature_cache = _precompute_cnn_features(
            model,
            train_samples + val_samples,
            eval_tf,
            device,
        )
        train_ds = PrecomputedFeatureDataset(train_samples, feature_cache)
        val_ds = PrecomputedFeatureDataset(val_samples, feature_cache)
    else:
        train_ds = FatigueSequenceDataset(train_samples, _build_transforms(train=True))
        val_ds = FatigueSequenceDataset(val_samples, _build_transforms(train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    history = []
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for seqs, labels in train_loader:
            seqs = seqs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model.forward_from_features(seqs) if use_precomputed else model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        scheduler.step()
        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)
        val = evaluate(model, val_loader, criterion, device, precomputed=use_precomputed)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["accuracy"],
            "val_f1": val["f1_drowsy"],
        })

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.3f} "
            f"val_f1={val['f1_drowsy']:.3f}"
        )

        if val["f1_drowsy"] > best_val_f1:
            best_val_f1 = val["f1_drowsy"]
            torch.save(model.state_dict(), model_out)
            print(f"[modern.train] Saved best model to {model_out}")

    _write_history(history, args.history_out)

    best_model = FatigueNet(pretrained=False).to(device)
    best_model.load_state_dict(torch.load(model_out, map_location=device))
    final_val = evaluate(best_model, val_loader, criterion, device, precomputed=use_precomputed)
    _write_confusion_matrix(final_val["y_true"], final_val["y_pred"], args.confusion_out)

    metrics = {
        "model_path": str(model_out),
        "data_root": args.data_root,
        "n_train_sequences": len(train_samples),
        "n_val_sequences": len(val_samples),
        "seq_len": args.seq_len,
        "stride": args.stride,
        "epochs": args.epochs,
        "pretrained": bool(args.pretrained),
        "freeze_cnn": bool(args.freeze_cnn),
        "precompute_features": bool(use_precomputed),
        "accuracy": float(final_val["accuracy"]),
        "precision_drowsy": float(final_val["precision_drowsy"]),
        "recall_drowsy": float(final_val["recall_drowsy"]),
        "f1_drowsy": float(final_val["f1_drowsy"]),
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[modern.train] Saved history to {args.history_out}")
    print(f"[modern.train] Saved metrics to {args.metrics_out}")
    print(f"[modern.train] Saved confusion matrix to {args.confusion_out}")


if __name__ == "__main__":
    main()
