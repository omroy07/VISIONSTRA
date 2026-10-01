"""
Decide which hazard, if any, should be spoken right now.

Pure Python and clock-injectable, so every rule below is unit-tested without
a camera or a speech engine. Output is an ``Alert``; the caller decides how
to voice it (pyttsx3 on the server, speechSynthesis in the browser).

Rules, checked in order for the rank-1 hazard of a frame:

  1. Nothing is said while speech is already playing, for an empty frame,
     or when the top hazard's priority is not in ``speak_priorities``.
  2. Escalation: a priority higher than the last alert speaks at once.
  3. New HIGH object: a HIGH hazard with a different ``track_id`` that has
     not been announced recently speaks at once.
  4. Otherwise wait ``cooldown_s[last alert's priority]`` after the last
     alert, and ``same_object_repeat_s`` before repeating the same object
     at the same or a lower priority.

The decision never depends on the sentence text, so a distance that jitters
from 2.1 m to 1.9 m does not count as a new warning.
"""

import threading
import time
from dataclasses import dataclass

from .fields import detection_distance, detection_name, normalise_position
from .hazard_config import DEFAULT_CONFIG
from .risk import priority_level

_DIRECTION_WORDS = {"center": "ahead", "left": "on your left", "right": "on your right"}
_OPENERS = {"HIGH": "Warning", "MEDIUM": "Caution", "LOW": "Note"}


@dataclass(frozen=True)
class Alert:
    text: str           # sentence to speak
    priority: str       # HIGH / MEDIUM
    track_id: object    # object announced (None when untracked)
    trigger: str        # first | escalation | new_high_object | cooldown_elapsed
    time: float


def _spoken_distance(distance):
    if distance < 1:
        return "less than 1 meter"
    if distance < 10:
        value = round(distance * 2) / 2          # nearest half metre
    else:
        value = round(distance)
    text = f"{value:g}"
    return f"{text} meter" if text == "1" else f"{text} meters"


def build_alert_text(top, other_hazards=0):
    """
    "Warning: bicycle ahead, 2 meters, approaching fast. 1 more hazard nearby."
    """
    priority = top.get("priority", "LOW")
    name = (detection_name(top) or "object").lower()
    parts = [name]

    direction = _DIRECTION_WORDS.get(normalise_position(top.get("direction")))
    if direction:
        parts[0] = f"{name} {direction}"

    distance = detection_distance(top)
    if distance is not None:
        parts.append(_spoken_distance(distance))

    if top.get("movement") == "approaching":
        parts.append("approaching fast" if top.get("imminent") else "approaching")

    text = f"{_OPENERS.get(priority, 'Note')}: " + ", ".join(parts) + "."
    if other_hazards == 1:
        text += " 1 more hazard nearby."
    elif other_hazards > 1:
        text += f" {other_hazards} more hazards nearby."
    return text


class AlertPolicy:

    def __init__(self, config=None, clock=time.monotonic):
        self._cfg = config or DEFAULT_CONFIG.alerts
        self._clock = clock
        self._lock = threading.Lock()
        self._last = None          # last Alert
        self._by_track = {}        # track_id -> (time, priority level)

    def reset(self):
        with self._lock:
            self._last = None
            self._by_track.clear()

    def consider(self, ranked, now=None, speaking=False):
        """Return the Alert to speak for this frame, or None."""
        if speaking or not ranked:
            return None
        cfg = self._cfg
        top = ranked[0]
        priority = top.get("priority")
        if priority not in cfg.speak_priorities:
            return None

        now = self._clock() if now is None else now
        level = priority_level(priority)
        track_id = top.get("track_id")

        with self._lock:
            trigger = self._trigger(priority, level, track_id, now)
            if trigger is None:
                return None

            others = 0
            if cfg.mention_other_hazards:
                others = sum(1 for det in ranked[1:]
                             if det.get("priority") in cfg.speak_priorities)
            alert = Alert(build_alert_text(top, others), priority, track_id, trigger, now)
            self._last = alert
            if track_id is not None:
                self._by_track[track_id] = (now, level)
            self._forget_old(now)
            return alert

    def _trigger(self, priority, level, track_id, now):
        cfg = self._cfg
        last = self._last
        if last is None:
            return "first"
        if level > priority_level(last.priority):
            return "escalation"

        previous = self._by_track.get(track_id) if track_id is not None else None
        recently_announced = (previous is not None
                              and now - previous[0] < cfg.same_object_repeat_s)
        if (cfg.new_high_object_interrupts and priority == "HIGH"
                and track_id is not None and track_id != last.track_id
                and not recently_announced):
            return "new_high_object"

        if now - last.time < cfg.cooldown_s.get(last.priority, 0):
            return None
        if recently_announced and level <= previous[1]:
            return None
        return "cooldown_elapsed"

    def _forget_old(self, now):
        horizon = self._cfg.same_object_repeat_s
        for track_id in [t for t, (at, _) in self._by_track.items() if now - at > horizon]:
            del self._by_track[track_id]
