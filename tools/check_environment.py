"""
Quick project readiness check.

Run:
    python tools/check_environment.py
"""

from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from data.frame_dataset import binary_counts, class_counts, list_fatigue_frames


MODULES = [
    "cv2",
    "mediapipe",
    "numpy",
    "scipy",
    "torch",
    "torchvision",
    "sklearn",
    "joblib",
    "pygame",
    "PIL",
]


def check_modules():
    print("Dependencies")
    for name in MODULES:
        code = (
            "import importlib; "
            f"m = importlib.import_module('{name}'); "
            "print(getattr(m, '__version__', 'installed'))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            version = lines[-1] if lines else "installed"
            print(f"  OK      {name}: {version}")
        else:
            lines = [
                line.strip()
                for line in (result.stderr + result.stdout).splitlines()
                if line.strip()
            ]
            message = lines[-1] if lines else "import failed"
            print(f"  MISSING {name}: {message}")


def check_data_and_models():
    print("\nDataset")
    records = list_fatigue_frames("frames/fatigue")
    print(f"  fatigue frames: {len(records)}")
    print(f"  per folder:     {class_counts(records)}")
    print(f"  binary:         {binary_counts(records)}")

    print("\nModels")
    for path in [config.CLASSIFIER_PATH, config.MODEL_PATH]:
        exists = Path(path).exists()
        print(f"  {'OK     ' if exists else 'MISSING'} {path}")

    print("\nTools")
    git_lfs = shutil.which("git-lfs")
    print(f"  {'OK     ' if git_lfs else 'MISSING'} git-lfs{f' ({git_lfs})' if git_lfs else ''}")


def main():
    check_modules()
    check_data_and_models()


if __name__ == "__main__":
    main()
