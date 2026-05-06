# Driver Fatigue Detection With Gesture-Based Activation

Computer vision coursework project for detecting simulated driver fatigue inside a parked vehicle. The system starts inactive, waits for a gesture sequence, then monitors the driver's face for fatigue cues.

## Current System

- Gesture activation: `OPEN_PALM -> THUMBS_UP` within 5 seconds.
- Classical fatigue pipeline: MediaPipe Face Mesh, EAR, MAR, head pose, blink-rate features, and threshold alerts.
- Classical ML pipeline: SVM + Random Forest ensemble trained from landmark features.
- Modern pipeline: MobileNetV2 + LSTM sequence model.
- Fusion: combines classical and modern scores when both trained models are available.
- UI: live OpenCV overlay with activation state, fatigue metrics, alert tags, score, and sound.

## Dataset Layout

The collected final frames are stored here:

```text
frames/fatigue/alert
frames/fatigue/eyes_closed
frames/fatigue/yawning
frames/fatigue/head_down
frames/gestures/open_palm
frames/gestures/thumbs_up
frames/gestures/no_gesture
frames/gestures/sequence_correct
frames/gestures/sequence_wrong
```

For fatigue training, labels are binary:

- `alert` -> `0`
- `eyes_closed`, `yawning`, `head_down` -> `1`

The shared mapper is [data/frame_dataset.py](data/frame_dataset.py).

## Setup

Use a clean virtual environment. This avoids the common NumPy/Torch/MediaPipe conflicts that happen in large Conda base environments.

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Trained weights and large files (Git LFS)

This repo uses **Git LFS** for images, model weights (`*.joblib`, `*.pth`), and the optional demo movie. After a fresh `git clone`, run **LFS** or you will only get **tiny pointer files** — **inference will fail** until the real blobs are downloaded.

```bash
brew install git-lfs   # or your OS package manager
git lfs install
git lfs pull
```

After `git lfs pull`, **`models/classical_classifier.joblib`** and **`models/fatigue_net.pth`** are the full trained artifacts. You can run the live system **immediately** with no training. Use the training sections below only to **reproduce** or **refresh** experiments.

The **`frames/`** JPEGs are also on LFS (~4 GB total). You only need them for **training** or **dataset inspection**; the **webcam demo does not read `frames/`** at runtime.

Check readiness:

```bash
python tools/check_environment.py
pytest -q -p no:capture
```

On some macOS Python builds, pytest's default capture plugin can crash inside a readline workaround. The `-p no:capture` flag avoids that environment issue and still runs the tests normally.

## Train The Classical ML Model

This extracts MediaPipe face landmarks from the collected frames, computes `[EAR, MAR, pitch, yaw, roll, blink_rate]`, trains the SVM + Random Forest ensemble, saves the model, and writes evaluation files.

```bash
python -m classical.train --data-root frames/fatigue --refresh-features
```

Outputs:

```text
models/classical_classifier.joblib
results/classical_features.csv
results/classical_metrics.json
results/classical_confusion_matrix.csv
```

For a quick smoke test:

```bash
python -m classical.train --data-root frames/fatigue --max-per-label 50 --refresh-features
```

## Train The Modern CNN-LSTM Model

Quick coursework run:

```bash
python -m modern.train --data-root frames/fatigue --epochs 3 --stride 10 --freeze-cnn
```

Stronger run:

```bash
python -m modern.train --data-root frames/fatigue --epochs 15 --stride 3 --pretrained --freeze-cnn --precompute-features --batch-size 32
```

Outputs:

```text
models/fatigue_net.pth
results/modern_training_history.csv
results/modern_metrics.json
results/modern_confusion_matrix.csv
```

If pretrained MobileNet weights are unavailable, the model falls back to random initialization.
The `--precompute-features` option caches the frozen MobileNet features once, then trains the LSTM much faster.

## Run The Live System

Requires a working webcam, **`git lfs pull`** so `models/` contains real weights, and an activated venv (see Setup).

```bash
python main.py
```

### Debug mode (`--debug`)

Use this for **demos, screen recordings, and understanding why an alert fires**. It overlays a panel with:

| Label | Meaning |
|-------|--------|
| **Thr** | Rule-only fatigue score (eyes / yawn / head streaks) before ML blending. |
| **ML** | Classical ensemble (SVM + RF) **drowsy probability** from landmark features. |
| **Cls** | `0.5×Thr + 0.5×ML` fed into fusion. |
| **Mod** | CNN-LSTM drowsy prob, or **`buf i/n`** while the temporal buffer fills (`n` = `FRAME_BUFFER_SIZE` in `config.py`), or **off** if weights are missing. |
| **Fin** | Fused score (classical + modern weights in `config.py`). |
| **Alert text** | **`RULE:…`** if a threshold tag fired; **`SCORE≥…`** if fused score exceeded **`FUSION_ALERT_THRESHOLD`**; **`RULE+SCORE`** if both; **`---`** if neither. |
| **Yaw** | Head pose yaw (useful for offset / in-car cameras). |

```bash
python main.py --debug
# equivalent:
VISION_DEBUG=1 python main.py
```

With debug on, the terminal prints one summary line every **`DEBUG_LOG_EVERY_FRAMES`** frames (default **15** in `config.py`).

Controls:

- `q`: quit
- `r`: reset activation

Flow:

1. The system starts inactive.
2. Show open palm.
3. Within 5 seconds, show thumbs up.
4. Fatigue detection activates.
5. Simulate fatigue: close eyes, yawn, nod head, or tilt head.

### Camera placement (in-car)

Use the **same fixed mount** for the demo as for **`frames/fatigue`** collection (parked vehicle, dash / mirror style offset — not necessarily straight-on webcam). Pose logic **re-baselines** pitch/roll for a short window after activation; large absolute **Yaw** on the debug line is typical from an angled rig. If **HEAD\_TILT** is still too sensitive off-axis, try raising **`HEAD_ROLL_THRESHOLD`** or **`HEAD_CONSEC_FRAMES`** in `config.py`.

## Demo Video Requirements

The final video must show:

- One group member seated in the driver's position in a parked car.
- System initially inactive.
- Correct `OPEN_PALM -> THUMBS_UP` gesture sequence.
- Activation of fatigue detection.
- Real-time simulated fatigue detection.
- Clearly visible face.

Suggested recording flow:

1. Park the vehicle safely and sit in the driver's position.
2. Start screen/video recording.
3. Run:

   ```bash
   source .venv/bin/activate
   python main.py
   ```

   Use `python main.py --debug` instead if you want the on-screen debug panel (Thr / ML / Mod / Fin) for the recording.

4. Show that the application starts in the `INACTIVE` state.
5. Keep hands neutral briefly to show that fatigue detection is not active yet.
6. Perform `OPEN_PALM`.
7. Within 5 seconds, perform `THUMBS_UP`.
8. Show that the UI changes to active monitoring.
9. Simulate fatigue by closing eyes, yawning, nodding down, or tilting the head.
10. Keep the fatigue alert visible long enough for the evaluator to read it.
11. Press `q` to quit.

## Report

The final report source is [REPORT_ACII2023.tex](REPORT_ACII2023.tex), formatted with the ACII 2023 / IEEE conference style. The compiled report is:

```text
REPORT_ACII2023.pdf
```

If the report changes, edit `REPORT_ACII2023.tex` and recompile with:

```bash
pdflatex -interaction=nonstopmode REPORT_ACII2023.tex
pdflatex -interaction=nonstopmode REPORT_ACII2023.tex
```

`IEEEtran.cls` is included locally because some TeX installations do not ship it by default.

The report figures are generated from the saved result files and stored in:

```text
results/figures/confusion_matrices.png
results/figures/modern_training_history.png
```

The underlying metric files are:

```text
results/classical_metrics.json
results/modern_metrics.json
results/classical_confusion_matrix.csv
results/modern_confusion_matrix.csv
```

The report contains the required sections: abstract, introduction, methodology, experimental setup, results, discussion, limitations, ethical impact statement, and conclusion.

## Packaging for submission (zip)

The **`frames/`** dataset is ~**4 GB**; full project zips are too large for many LMS tools.

- **Recommended (small zip, ~20 MB):** omits `frames/` and `demo_video.mov`. Includes code, `models/`, report, results, tests, and **`SUBMISSION.md`** (how to get data / video).

```bash
bash scripts/make_submission_zip.sh slim
# -> vision-group-submission-slim.zip
```

- **Full mirror of the working tree** (no `.git` / `.venv`; large):

```bash
bash scripts/make_submission_zip.sh full
# -> vision-group-submission-full.zip
```

Upload **`demo_video.mov`** separately if the platform has a dedicated video slot or a file-size cap.

## Troubleshooting Notes

- Use `.venv`, not the Conda base environment, to avoid NumPy/Torch/MediaPipe version conflicts.
- If `git status` or pulling image files fails, install Git LFS with `brew install git-lfs`, then run `git lfs install` and `git lfs pull`.
- If `pytest -q` crashes on macOS, use `pytest -q -p no:capture`.
- If LaTeX creates `.aux`, `.log`, or `missfont.log` files, those are temporary build files and can be deleted.
