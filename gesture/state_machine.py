import time
import config


class GestureStateMachine:
    """
    Tracks a multi-gesture activation sequence.

    States:
        INACTIVE           – waiting for first gesture
        GESTURE_1_SEEN     – first gesture seen, waiting for second within window
        ACTIVE             – sequence completed, fatigue detection enabled
        DEACTIVATE_G1_SEEN – deactivation first gesture seen, waiting for second
    """

    _STATE_INACTIVE  = "INACTIVE"
    _STATE_G1_SEEN   = "GESTURE_1_SEEN"
    _STATE_ACTIVE    = "ACTIVE"
    _STATE_DEACT_G1  = "DEACTIVATE_G1_SEEN"

    def __init__(self, sequence=None, window_seconds=None):
        self.sequence = sequence or config.GESTURE_SEQUENCE
        self.window   = window_seconds or config.GESTURE_WINDOW_SECONDS
        self._state   = self._STATE_INACTIVE
        self._t0      = None   # timestamp when G1 was seen

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, gesture_label):
        """
        Feed a gesture label (or None) for the current frame.
        Returns the new state string.
        """
        if self._state == self._STATE_ACTIVE:
            if gesture_label == self.sequence[0]:
                self._state = self._STATE_DEACT_G1
                self._t0 = time.time()
            return self._state

        if self._state == self._STATE_DEACT_G1:
            elapsed = time.time() - self._t0

            if elapsed > self.window:
                # Window expired — return to ACTIVE
                self._state = self._STATE_ACTIVE
                self._t0 = None
            elif gesture_label == self.sequence[1]:
                self._reset()
            elif gesture_label == self.sequence[0]:
                # Repeat of G1 — refresh the timer
                self._t0 = time.time()

            return self._state

        if self._state == self._STATE_INACTIVE:
            if gesture_label == self.sequence[0]:
                self._state = self._STATE_G1_SEEN
                self._t0 = time.time()

        elif self._state == self._STATE_G1_SEEN:
            elapsed = time.time() - self._t0

            if elapsed > self.window:
                # Window expired — reset
                self._reset()
            elif gesture_label == self.sequence[1]:
                self._state = self._STATE_ACTIVE
            elif gesture_label == self.sequence[0]:
                # Repeat of G1 — refresh the timer
                self._t0 = time.time()

        return self._state

    def is_active(self):
        return self._state == self._STATE_ACTIVE

    def time_remaining(self):
        """Seconds left in the gesture window, or None if not in a timed state."""
        if self._state not in (self._STATE_G1_SEEN, self._STATE_DEACT_G1):
            return None
        remaining = self.window - (time.time() - self._t0)
        return max(0.0, remaining)

    def reset(self):
        """Manually deactivate (e.g. for testing or logout)."""
        self._reset()

    @property
    def state(self):
        return self._state

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reset(self):
        self._state = self._STATE_INACTIVE
        self._t0 = None
