import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from classical.features import compute_ear, compute_mar, compute_head_pose
from classical.landmarks import LEFT_EYE, RIGHT_EYE, MOUTH_OUTER


def _make_landmarks(n=468):
    """Return zeroed landmark array."""
    return np.zeros((n, 2), dtype=np.float32)


# ── EAR tests ─────────────────────────────────────────────────────────────────

def _set_eye_open(lm, indices):
    """Place eye landmarks so EAR ≈ 0.35 (open eye)."""
    # p0=left, p3=right, p1/p5 top/bottom pairs, p2/p4 top/bottom pairs
    # Horizontal span = 60px, vertical span = ~21px → EAR ≈ 0.35
    p = lm.copy()
    p[indices[0]] = [0,  30]   # left corner
    p[indices[1]] = [20, 10]   # top-left
    p[indices[2]] = [40, 10]   # top-right
    p[indices[3]] = [60, 30]   # right corner
    p[indices[4]] = [40, 50]   # bottom-right
    p[indices[5]] = [20, 50]   # bottom-left
    return p


def _set_eye_closed(lm, indices):
    """Place eye landmarks so EAR ≈ 0.0 (closed eye)."""
    p = lm.copy()
    for idx in indices:
        p[idx] = [30, 30]   # all at the same point
    return p


def test_ear_open_eyes():
    lm = _make_landmarks()
    lm = _set_eye_open(lm, LEFT_EYE)
    lm = _set_eye_open(lm, RIGHT_EYE)
    ear = compute_ear(lm)
    assert ear > 0.25, f"Open-eye EAR should be > 0.25, got {ear:.4f}"


def test_ear_closed_eyes():
    lm = _make_landmarks()
    lm = _set_eye_closed(lm, LEFT_EYE)
    lm = _set_eye_closed(lm, RIGHT_EYE)
    ear = compute_ear(lm)
    assert ear < 0.05, f"Closed-eye EAR should be < 0.05, got {ear:.4f}"


def test_ear_returns_float():
    lm = _make_landmarks()
    assert isinstance(compute_ear(lm), float)


# ── MAR tests ─────────────────────────────────────────────────────────────────

def _set_mouth_open(lm):
    """Wide-open mouth → MAR > 0.6."""
    m = MOUTH_OUTER
    p = lm.copy()
    # Horizontal width = 80px
    p[m["left_corner"][0]]  = [10, 100]
    p[m["right_corner"][0]] = [90, 100]
    # Vertical gaps ≈ 60px each
    p[m["top_left"][0]]     = [30,  70]
    p[m["bottom_left"][0]]  = [30, 130]
    p[m["top_inner"][0]]    = [50,  65]
    p[m["bottom_inner"][0]] = [50, 135]
    p[m["top_right"][0]]    = [70,  70]
    p[m["bottom_right"][0]] = [70, 130]
    return p


def _set_mouth_closed(lm):
    """Closed mouth → MAR < 0.2."""
    m = MOUTH_OUTER
    p = lm.copy()
    p[m["left_corner"][0]]  = [10, 100]
    p[m["right_corner"][0]] = [90, 100]
    p[m["top_left"][0]]     = [30,  98]
    p[m["bottom_left"][0]]  = [30, 102]
    p[m["top_inner"][0]]    = [50,  98]
    p[m["bottom_inner"][0]] = [50, 102]
    p[m["top_right"][0]]    = [70,  98]
    p[m["bottom_right"][0]] = [70, 102]
    return p


def test_mar_open_mouth():
    lm = _set_mouth_open(_make_landmarks())
    mar = compute_mar(lm)
    assert mar > 0.6, f"Open-mouth MAR should be > 0.6, got {mar:.4f}"


def test_mar_closed_mouth():
    lm = _set_mouth_closed(_make_landmarks())
    mar = compute_mar(lm)
    assert mar < 0.2, f"Closed-mouth MAR should be < 0.2, got {mar:.4f}"


def test_mar_returns_float():
    lm = _make_landmarks()
    assert isinstance(compute_mar(lm), float)


# ── Head pose tests ───────────────────────────────────────────────────────────

def test_head_pose_returns_three_floats():
    lm = _make_landmarks()
    result = compute_head_pose(lm, (480, 640))
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


def test_head_pose_no_crash_on_degenerate_input():
    """solvePnP should not raise even with degenerate zero landmarks."""
    lm = _make_landmarks()
    pitch, yaw, roll = compute_head_pose(lm, (480, 640))
    assert all(isinstance(v, float) for v in (pitch, yaw, roll))
