# =============================================================================
# PLACEHOLDER — run this script once your own dataset is collected.
#
# Prerequisites:
#   1. Run data/collect.py inside the car to capture labeled video clips.
#   2. Run data/preprocess.py to extract frames into:
#          data/frames/alert/   (frame_0001.jpg, ...)
#          data/frames/drowsy/  (frame_0001.jpg, ...)
#   3. Run:  python modern/train.py
#   4. Best weights saved to models/fatigue_net.pth
# =============================================================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import config
from modern.model import FatigueNet

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT    = "data/frames"
EPOCHS       = 30
BATCH_SIZE   = 8
LR           = 1e-4
SEQ_LEN      = config.FRAME_BUFFER_SIZE
CROP_SIZE    = config.FACE_CROP_SIZE
VAL_SPLIT    = 0.2
SEED         = 42

torch.manual_seed(SEED)
random.seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────

class FatigueVideoDataset(Dataset):
    """
    Loads consecutive frame sequences from data/frames/{alert,drowsy}/.
    Each sample is a (SEQ_LEN, C, H, W) tensor and a label (0=alert, 1=drowsy).
    """

    def __init__(self, root, seq_len, transform=None):
        self.seq_len   = seq_len
        self.transform = transform
        self.samples   = []   # list of (list_of_frame_paths, label)

        for label_idx, class_name in enumerate(["alert", "drowsy"]):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            frames = sorted([
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".png"))
            ])
            # Sliding window sequences
            for i in range(len(frames) - seq_len + 1):
                self.samples.append((frames[i:i + seq_len], label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        return torch.stack(frames), label   # (T, C, H, W), int


def _get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    train_tf, val_tf = _get_transforms()

    full_dataset = FatigueVideoDataset(DATA_ROOT, SEQ_LEN, transform=train_tf)
    if len(full_dataset) == 0:
        raise RuntimeError(
            f"No data found in '{DATA_ROOT}'. "
            "Run data/collect.py and data/preprocess.py first."
        )

    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset = FatigueVideoDataset(DATA_ROOT, SEQ_LEN, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = FatigueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                logits = model(seqs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += len(labels)

        train_acc = correct / total
        val_acc   = val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"loss={train_loss/total:.4f}  "
              f"train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")

    print(f"\n[Train] Done. Best val acc: {best_val_acc:.3f}")
    print(f"[Train] Weights saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    train()
