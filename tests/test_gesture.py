import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pytest
from unittest.mock import patch
from gesture.state_machine import GestureStateMachine


SEQUENCE = ["OPEN_PALM", "THUMBS_UP"]
WINDOW   = 5.0


def _make_sm():
    return GestureStateMachine(sequence=SEQUENCE, window_seconds=WINDOW)


# ── Correct sequence ──────────────────────────────────────────────────────────

def test_correct_sequence_activates():
    sm = _make_sm()
    sm.update("OPEN_PALM")
    sm.update("THUMBS_UP")
    assert sm.is_active()


def test_state_after_full_sequence():
    sm = _make_sm()
    sm.update("OPEN_PALM")
    sm.update("THUMBS_UP")
    assert sm.state == "ACTIVE"


# ── Wrong order ───────────────────────────────────────────────────────────────

def test_wrong_order_stays_inactive():
    sm = _make_sm()
    sm.update("THUMBS_UP")   # G2 first — wrong
    sm.update("OPEN_PALM")
    assert not sm.is_active()


def test_wrong_second_gesture_stays_in_g1_seen():
    sm = _make_sm()
    sm.update("OPEN_PALM")
    sm.update(None)           # no gesture
    assert sm.state == "GESTURE_1_SEEN"


# ── Timeout ───────────────────────────────────────────────────────────────────

def test_timeout_resets_to_inactive():
    sm = _make_sm()
    sm.update("OPEN_PALM")
    # Fast-forward time past the window
    with patch("gesture.state_machine.time") as mock_time:
        mock_time.time.return_value = time.time() + WINDOW + 1
        sm.update(None)
    assert sm.state == "INACTIVE"
    assert not sm.is_active()


# ── Timer refresh ─────────────────────────────────────────────────────────────

def test_repeated_g1_refreshes_timer():
    sm = _make_sm()
    sm.update("OPEN_PALM")
    t0 = sm._t0
    time.sleep(0.05)
    sm.update("OPEN_PALM")   # repeated G1 should refresh timer
    assert sm._t0 >= t0


# ── Manual reset ──────────────────────────────────────────────────────────────

def test_manual_reset():
    sm = _make_sm()
    sm.update("OPEN_PALM")
    sm.update("THUMBS_UP")
    assert sm.is_active()
    sm.reset()
    assert sm.state == "INACTIVE"
    assert not sm.is_active()


# ── Time remaining ────────────────────────────────────────────────────────────

def test_time_remaining_none_when_inactive():
    sm = _make_sm()
    assert sm.time_remaining() is None


def test_time_remaining_positive_after_g1():
    sm = _make_sm()
    sm.update("OPEN_PALM")
    remaining = sm.time_remaining()
    assert remaining is not None
    assert 0 < remaining <= WINDOW


def test_time_remaining_none_when_active():
    sm = _make_sm()
    sm.update("OPEN_PALM")
    sm.update("THUMBS_UP")
    assert sm.time_remaining() is None


# ── No gesture input ─────────────────────────────────────────────────────────

def test_none_input_doesnt_change_inactive_state():
    sm = _make_sm()
    sm.update(None)
    assert sm.state == "INACTIVE"


# ── Deactivation sequence ─────────────────────────────────────────────────────

def test_deactivation_sequence():
    """Full deactivation sequence while ACTIVE → goes INACTIVE."""
    sm = _make_sm()
    sm.update("OPEN_PALM")
    sm.update("THUMBS_UP")
    assert sm.is_active()
    sm.update("OPEN_PALM")
    sm.update("THUMBS_UP")
    assert sm.state == "INACTIVE"
    assert not sm.is_active()


def test_deactivate_g1_seen_state():
    """After G1 while active, state is DEACTIVATE_G1_SEEN."""
    sm = _make_sm()
    sm.update("OPEN_PALM")
    sm.update("THUMBS_UP")
    assert sm.is_active()
    sm.update("OPEN_PALM")
    assert sm.state == "DEACTIVATE_G1_SEEN"


def test_deactivation_wrong_order():
    """Wrong second gesture while ACTIVE → stays ACTIVE (not deactivated)."""
    sm = _make_sm()
    sm.update("OPEN_PALM")
    sm.update("THUMBS_UP")
    assert sm.is_active()
    sm.update("OPEN_PALM")
    sm.update("OPEN_PALM")   # repeating G1 instead of G2 — refreshes timer
    assert sm.state == "DEACTIVATE_G1_SEEN"
    # Sending an unrelated gesture does not deactivate
    sm2 = _make_sm()
    sm2.update("OPEN_PALM")
    sm2.update("THUMBS_UP")
    sm2.update("OPEN_PALM")
    sm2.update(None)         # no gesture — stays in deact G1
    assert sm2.state == "DEACTIVATE_G1_SEEN"


def test_deactivation_timeout():
    """Deactivation G1 seen but window expires → returns to ACTIVE."""
    sm = _make_sm()
    sm.update("OPEN_PALM")
    sm.update("THUMBS_UP")
    assert sm.is_active()
    sm.update("OPEN_PALM")
    assert sm.state == "DEACTIVATE_G1_SEEN"
    with patch("gesture.state_machine.time") as mock_time:
        mock_time.time.return_value = time.time() + WINDOW + 1
        sm.update(None)
    assert sm.state == "ACTIVE"
    assert sm.is_active()


def test_time_remaining_during_deactivation():
    """time_remaining() returns a value during DEACTIVATE_G1_SEEN."""
    sm = _make_sm()
    sm.update("OPEN_PALM")
    sm.update("THUMBS_UP")
    sm.update("OPEN_PALM")
    remaining = sm.time_remaining()
    assert remaining is not None
    assert 0 < remaining <= WINDOW
