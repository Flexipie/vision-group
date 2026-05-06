#!/usr/bin/env bash
# Build a coursework submission zip from the repo root (no .git / .venv / caches).
#
#   bash scripts/make_submission_zip.sh slim [out.zip]
#       Omit frames/ (~3.9GB) and demo_video.mov (~110MB). Use for LMS size limits.
#       Includes SUBMISSION.md explaining how to obtain data / video.
#
#   bash scripts/make_submission_zip.sh full [out.zip]
#       Everything except dev junk (same as before; ~4GB if frames + video present).
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-}"
OUT="${2:-}"
if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  echo "Usage: bash scripts/make_submission_zip.sh {slim|full} [output.zip]"
  echo "  slim  -> default: vision-group-submission-slim.zip"
  echo "  full  -> default: vision-group-submission-full.zip"
  exit 0
fi

case "$MODE" in
  slim)
    OUT="${OUT:-vision-group-submission-slim.zip}"
    EXCLUDE_FRAMES=1
    EXCLUDE_DEMO=1
    ;;
  full)
    OUT="${OUT:-vision-group-submission-full.zip}"
    EXCLUDE_FRAMES=0
    EXCLUDE_DEMO=0
    ;;
  *)
    echo "Unknown mode: $MODE (use slim or full)" >&2
    exit 1
    ;;
esac

OUT_ABS="$ROOT/$OUT"
rm -f "$OUT_ABS"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
STAGE_ROOT="$STAGE/vision-group"

echo "Staging under $STAGE_ROOT (mode=$MODE)..."

RSYNC_EXCLUDES=(
  -a
  --exclude '.git/'
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '*/__pycache__/'
  --exclude '.pytest_cache/'
  --exclude '*.pyc'
  --exclude '.DS_Store'
  --exclude 'REPORT_ACII2023.aux'
  --exclude 'REPORT_ACII2023.log'
  --exclude 'vision-group-submission-slim.zip'
  --exclude 'vision-group-submission-full.zip'
  --exclude 'vision-group-submission.zip'
)
if [[ "$EXCLUDE_FRAMES" -eq 1 ]]; then
  RSYNC_EXCLUDES+=(--exclude 'frames/')
  echo "  (excluding frames/ to shrink zip — see SUBMISSION.md)"
fi
if [[ "$EXCLUDE_DEMO" -eq 1 ]]; then
  RSYNC_EXCLUDES+=(--exclude 'demo_video.mov')
  echo "  (excluding demo_video.mov — upload video to LMS separately if required)"
fi

rsync "${RSYNC_EXCLUDES[@]}" "$ROOT/" "$STAGE_ROOT/"

if [[ "$EXCLUDE_FRAMES" -eq 1 ]]; then
  cat > "$STAGE_ROOT/SUBMISSION.md" << 'EOF'
# Submission package (slim archive)

This zip omits **`frames/`** (in-car JPEG dataset, ~4 GB) and **`demo_video.mov`** so it fits typical LMS limits.

Included: full **source code**, **`requirements.txt`**, **`README.md`**, **trained weights** in `models/`, **report** (`REPORT_ACII2023.tex` / `.pdf`), **results** metrics and figures, **tests**, and **`IEEEtran.cls`**.

## Dataset

- If you use **Git / GitHub**: clone the group repo and run **`git lfs pull`** so `frames/fatigue/` and `frames/gestures/` download.

- **Retraining** (optional): after LFS pull, see README: `python -m classical.train ...`, `python -m modern.train ...`

## Demo video

Submit **`demo_video.mov`** via the course platform separately if the assignment requires the video outside this zip.
EOF
fi

(
  cd "$STAGE"
  zip -r -q "$OUT_ABS" vision-group
)

echo "Wrote: $OUT_ABS"
ls -lh "$OUT_ABS"
