"""
One front door for a camera stream: raw detections in, ranked hazards out.

    detections ──► IouTracker ──► MotionEstimator ──► risk.score_detections
                   (track_id)     (smoothed distance,   (score, priority,
                                   movement, TTC)        reason, breakdown)
                                                          │
                     ranked list ◄── risk.sort_and_rank ◄─┴─ priority hold

Keep one ``HazardPrioritizer`` per camera stream: it remembers the previous
frames. ``risk.rank_detections`` stays available for one-off, stateless use.

Priority hold: a tracked object's priority may rise at once but only falls
after it has stayed lower for ``stability.downgrade_hold_s`` seconds. A held
item keeps its real ``risk_score``, reports ``raw_priority`` and
``priority_held: true``, and says so in its reason.
"""

import time

from .fields import raw_distance
from .hazard_config import DEFAULT_CONFIG
from .motion import MotionEstimator
from .risk import priority_level, score_detections, sort_and_rank
from .tracking import IouTracker


class HazardPrioritizer:

    def __init__(self, config=None, clock=time.monotonic):
        self.config = config or DEFAULT_CONFIG
        self._clock = clock
        self._tracker = IouTracker(self.config.tracking)
        self._motion = MotionEstimator(self.config.motion)
        self._held = {}      # track_id -> (priority, time it was last confirmed)

    def reset(self):
        self._tracker.reset()
        self._motion.reset()
        self._held.clear()

    def process(self, detections, now=None):
        """Return the frame's detections scored, stabilised and ranked."""
        now = self._clock() if now is None else now

        tracked = self._tracker.update(detections, now)
        for det in tracked:
            self._add_motion(det, now)

        scored = score_detections(tracked, self.config.scoring)
        for det in scored:
            self._hold_priority(det, now)

        active = self._tracker.active_ids
        self._motion.forget_except(active)
        for track_id in list(self._held):
            if track_id not in active:
                del self._held[track_id]

        return sort_and_rank(scored)

    def _add_motion(self, det, now):
        reading = self._motion.update(det.get("track_id"), raw_distance(det), now)
        if reading.smoothed_distance_m is not None:
            det["smoothed_distance_m"] = reading.smoothed_distance_m
        # A caller that already knows the movement (e.g. from a sensor) wins.
        if det.get("movement") is None and reading.movement is not None:
            det["movement"] = reading.movement
            det["closing_speed_mps"] = reading.closing_speed_mps
            if reading.time_to_contact_s is not None:
                det["time_to_contact_s"] = reading.time_to_contact_s

    def _hold_priority(self, det, now):
        det["raw_priority"] = det["priority"]
        det["priority_held"] = False
        track_id = det.get("track_id")
        if track_id is None:
            return

        held = self._held.get(track_id)
        current = det["priority"]
        if held is None or priority_level(current) >= priority_level(held[0]):
            self._held[track_id] = (current, now)
            return

        held_priority, confirmed_at = held
        if now - confirmed_at < self.config.stability.downgrade_hold_s:
            det["priority"] = held_priority
            det["priority_held"] = True
            det["reason"] += (f" Held at {held_priority} briefly so the"
                              f" warning does not flicker (score alone gives {current}).")
        else:
            self._held[track_id] = (current, now)
