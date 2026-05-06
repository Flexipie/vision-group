# Rubric Verification

Pre-submission check of the system against `Project idea 2 - Driver Fatigue
Detection with Gesture-Based Activation` and the course rule that requires
a per-member contributions section in the report.

## Gesture-Based Activation

- The system starts in `INACTIVE`. Fatigue detection in [main.py](main.py)
  only runs when `state_machine.is_active()` returns `True`.
- The activation sequence is `OPEN_PALM -> THUMBS_UP`, defined in
  [config.py](config.py) (`GESTURE_SEQUENCE`, `GESTURE_WINDOW_SECONDS`),
  with a 5 second window. Two different gestures, correct order, time
  window: covered.
- Wrong order or timeout returns the state machine to `INACTIVE`.
- Behavior covered by [tests/test_gesture.py](tests/test_gesture.py):
  `test_wrong_order_stays_inactive`, `test_timeout_resets_to_inactive`,
  `test_wrong_second_gesture_stays_in_g1_seen`.

## Fatigue Detection

- Eye closure: EAR via `compute_ear` in
  [classical/features.py](classical/features.py), threshold logic in
  [classical/detector.py](classical/detector.py).
- Yawning: MAR via `compute_mar`.
- Head pose: pitch and roll from `cv2.solvePnP` via `compute_head_pose`,
  measured against a per-user baseline so seated posture does not trigger
  alerts.
- Classical ML: SVM + Random Forest soft-voting ensemble at
  `models/classical_classifier.joblib`.
- Modern: MobileNetV2 + LSTM at `models/fatigue_net.pth`.
- Fusion: [fusion/combiner.py](fusion/combiner.py) blends classical and
  modern scores and falls back to classical-only when the modern model is
  unavailable.

## Data Acquisition

- All frames in `frames/fatigue/` and `frames/gestures/` were captured
  inside a stationary vehicle with the subject in the driver position.
- MobileNetV2 ImageNet weights are used only for feature extraction, which
  is allowed by the brief.

## Mandatory Deliverables

- Source code: organized by `gesture/`, `classical/`, `modern/`, `fusion/`,
  `ui/`, `data/`, `tools/`, `tests/`. Module docstrings and section
  headers in each file. Running instructions in [README.md](README.md).
- Technical report: [REPORT_ACII2023.tex](REPORT_ACII2023.tex) and
  `REPORT_ACII2023.pdf` cover system description, methodology, experimental
  setup, results, discussion, limitations, ethical impact statement, and
  conclusion. The byline lists all six team members. The `Author
  Contributions` section after the conclusion satisfies the per-member
  contributions rule.
- Demonstration video: [demo_video.mov](demo_video.mov) at the repository
  root. Recording flow documented in the `Demo Video Requirements` section
  of `README.md`.

## Test Suite

Run from the project root, inside the project venv:

```bash
pytest -q -p no:capture
```

The `-p no:capture` flag is recommended in `README.md` for the macOS Python
builds used during development.

Coverage:

- [tests/test_features.py](tests/test_features.py): EAR, MAR, and head-pose
  feature computations on synthetic landmarks.
- [tests/test_gesture.py](tests/test_gesture.py): activation,
  deactivation, timeout, and timer-refresh paths of the gesture state
  machine.

## Items Flagged for Team Review

- [data/preprocess.py](data/preprocess.py) and
  [data/collect.py](data/collect.py) reference an older label set
  (`drowsy`, `yawn`, `head_nod`) and a `data/raw` workflow that does not
  match the active pipeline. The active pipeline uses
  `frames/fatigue/{alert,eyes_closed,yawning,head_down}` through
  [data/frame_dataset.py](data/frame_dataset.py). The live demo, the
  training commands, and the test suite do not depend on these scripts, so
  the submitted system runs correctly. Removal or update is left to the
  team's discretion.
