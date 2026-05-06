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
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Git LFS before cloning/pulling the dataset on a new machine:

```bash
brew install git-lfs
git lfs install
git lfs pull
```

Check readiness:

```bash
python tools/check_environment.py
pytest -q -p no:capture
```

On some macOS Python builds, pytest's default capture plugin can crash inside a readline workaround. The `-p no:capture` flag avoids that environment issue and still runs the tests normally.

The submitted model files are already included in `models/`, so training is not required just to run the demo. Retraining is only needed if the dataset changes or if you want to reproduce the reported metrics.

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

```bash
python main.py
```

Controls:

- `q`: quit
- `r`: reset activation

Flow:

1. The system starts inactive.
2. Show open palm.
3. Within 5 seconds, show thumbs up.
4. Fatigue detection activates.
5. Simulate fatigue: close eyes, yawn, nod head, or tilt head.

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

## Troubleshooting Notes

- Use `.venv`, not the Conda base environment, to avoid NumPy/Torch/MediaPipe version conflicts.
- If `git status` or pulling image files fails, install Git LFS with `brew install git-lfs`, then run `git lfs install` and `git lfs pull`.
- If `pytest -q` crashes on macOS, use `pytest -q -p no:capture`.
- If LaTeX creates `.aux`, `.log`, or `missfont.log` files, those are temporary build files and can be deleted.
