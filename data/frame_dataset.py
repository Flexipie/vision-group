"""
Shared helpers for using the collected frame dataset.

The repository stores final captured frames under:

    frames/fatigue/{alert,yawning,head_down,eyes_closed}

The training code uses binary labels:
    0 = alert / not fatigued
    1 = drowsy / fatigued
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


FATIGUE_LABEL_MAP: Dict[str, int] = {
    "alert": 0,
    "eyes_closed": 1,
    "yawning": 1,
    "head_down": 1,
    "drowsy": 1,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class FrameRecord:
    path: Path
    label: int
    label_name: str
    clip_id: str


def _clip_id(path: Path) -> str:
    """
    Derive a stable clip id from names like IMG_5725_00045.jpg.
    This keeps temporal windows from crossing source clips.
    """
    stem = path.stem
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]


def list_fatigue_frames(data_root: str | Path = "frames/fatigue") -> List[FrameRecord]:
    root = Path(data_root)
    records: List[FrameRecord] = []

    for label_name, label in FATIGUE_LABEL_MAP.items():
        label_dir = root / label_name
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    FrameRecord(
                        path=path,
                        label=label,
                        label_name=label_name,
                        clip_id=f"{label_name}:{_clip_id(path)}",
                    )
                )

    return records


def class_counts(records: Iterable[FrameRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        counts[record.label_name] = counts.get(record.label_name, 0) + 1
    return counts


def binary_counts(records: Iterable[FrameRecord]) -> Dict[str, int]:
    counts = {"alert": 0, "drowsy": 0}
    for record in records:
        counts["drowsy" if record.label == 1 else "alert"] += 1
    return counts


def limit_per_label(records: Sequence[FrameRecord], max_per_label: int | None) -> List[FrameRecord]:
    if max_per_label is None or max_per_label <= 0:
        return list(records)

    seen: Dict[str, int] = {}
    limited: List[FrameRecord] = []
    for record in records:
        count = seen.get(record.label_name, 0)
        if count < max_per_label:
            limited.append(record)
            seen[record.label_name] = count + 1
    return limited
