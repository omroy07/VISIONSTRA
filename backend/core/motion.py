"""
Estimate whether each tracked object is approaching, stationary or receding.

Monocular distance estimates jitter: for a person 12 m away a 3-pixel wobble
in box width is already ±1.2 m. Comparing two consecutive frames therefore
invents motion. Instead, per track:

  1. keep the raw distances from the last ``window_s`` seconds and fit a
     least-squares line through them; its slope is the closing speed;
  2. accept that trend only when it clearly stands out from the scatter
     around the line (|slope| / standard error >= ``min_t_stat``), the speed
     is at least ``min_speed_mps``, and the object would cover its distance
     within ``max_ttc_s`` seconds; otherwise the object is *stationary*;
  3. switch to *approaching* or *receding* only after that verdict has held
     for ``confirm_s`` seconds, so one unlucky frame cannot raise an alarm
     (falling back to *stationary* is immediate).

A separately smoothed distance (exponential moving average) is reported for
display, speech and scoring.

Everything is time-based, so behaviour is the same at 3 fps (browser API) and
30 fps (live camera). With too little history the movement stays unknown,
which the scorer treats as 0 points. Time-to-contact (smoothed distance ÷
closing speed) is reported for approaching objects; the scorer uses it for
the "imminent" rule.
"""

import math
from collections import deque
from dataclasses import dataclass

from .fields import as_positive_float
from .hazard_config import DEFAULT_CONFIG


@dataclass(frozen=True)
class MotionReading:
    smoothed_distance_m: float = None
    movement: str = None             # "approaching" | "stationary" | "receding" | None
    closing_speed_mps: float = None  # positive = getting closer
    time_to_contact_s: float = None  # only when approaching


def fit_trend(samples):
    """
    Least-squares line through (time, distance) samples.

    Returns (slope in m/s, standard error of the slope), or None when the
    samples do not span any time. The standard error is 0 for a perfect fit.
    """
    n = len(samples)
    mean_t = sum(t for t, _ in samples) / n
    mean_d = sum(d for _, d in samples) / n
    spread_t = sum((t - mean_t) ** 2 for t, _ in samples)
    if spread_t == 0:
        return None
    slope = sum((t - mean_t) * (d - mean_d) for t, d in samples) / spread_t
    if n < 3:
        return slope, math.inf
    intercept = mean_d - slope * mean_t
    residual = sum((d - (intercept + slope * t)) ** 2 for t, d in samples)
    return slope, math.sqrt(residual / (n - 2) / spread_t)


class MotionEstimator:

    def __init__(self, config=None):
        self._cfg = config or DEFAULT_CONFIG.motion
        self._tracks = {}

    def reset(self):
        self._tracks.clear()

    def forget_except(self, keep_ids):
        for track_id in list(self._tracks):
            if track_id not in keep_ids:
                del self._tracks[track_id]

    def update(self, track_id, distance, now):
        """Add one distance reading for ``track_id`` and return a MotionReading."""
        distance = as_positive_float(distance)
        if track_id is None or distance is None:
            return MotionReading()

        cfg = self._cfg
        state = self._tracks.get(track_id)
        if state is None:
            state = {"ema": distance, "samples": deque(),
                     "movement": None, "candidate": None, "candidate_since": None}
            self._tracks[track_id] = state
        else:
            state["ema"] += cfg.smoothing_alpha * (distance - state["ema"])

        samples = state["samples"]
        samples.append((now, distance))
        while samples and now - samples[0][0] > cfg.window_s:
            samples.popleft()

        smoothed = state["ema"]
        verdict, closing = self._verdict(samples, smoothed)
        movement = self._confirm(state, verdict, now)

        if movement is None or closing is None:
            return MotionReading(round(smoothed, 2), movement)
        if movement == "approaching" and closing > 0:
            return MotionReading(round(smoothed, 2), movement, round(closing, 2),
                                 round(smoothed / closing, 1))
        return MotionReading(round(smoothed, 2), movement, round(closing, 2))

    def _verdict(self, samples, smoothed):
        """This frame's evidence: (movement or None, closing speed)."""
        cfg = self._cfg
        if len(samples) < cfg.min_samples or samples[-1][0] - samples[0][0] < cfg.min_span_s:
            return None, None
        trend = fit_trend(samples)
        if trend is None:
            return None, None

        slope, std_error = trend
        closing = -slope
        speed = abs(closing)
        clear_trend = speed > 0 and (std_error == 0 or speed / std_error >= cfg.min_t_stat)
        if (clear_trend and speed >= cfg.min_speed_mps
                and smoothed / speed <= cfg.max_ttc_s):
            return ("approaching" if closing > 0 else "receding"), closing
        return "stationary", closing

    def _confirm(self, state, verdict, now):
        """Apply the ``confirm_s`` persistence rule and return the movement."""
        if verdict is None:
            return state["movement"]
        if verdict == "stationary" or verdict == state["movement"]:
            state["movement"] = verdict
            state["candidate"] = None
            return verdict
        if state["candidate"] != verdict:
            state["candidate"], state["candidate_since"] = verdict, now
        if now - state["candidate_since"] >= self._cfg.confirm_s:
            state["movement"] = verdict
            state["candidate"] = None
        return state["movement"]
