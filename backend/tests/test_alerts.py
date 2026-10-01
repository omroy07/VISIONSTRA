"""Which hazard is spoken, and when (backend/core/alerts.py)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.alerts import AlertPolicy, build_alert_text
from core.hazard_config import config_from_dict


def hazard(priority, track_id=1, name="car", distance=5.0, direction="Left", **extra):
    return {"object": name, "priority": priority, "track_id": track_id,
            "distance_m": distance, "direction": direction, **extra}


class TestWhenToSpeak(unittest.TestCase):

    def setUp(self):
        self.policy = AlertPolicy()

    def say(self, ranked, t, speaking=False):
        alert = self.policy.consider(ranked, now=t, speaking=speaking)
        return alert.trigger if alert else None

    def test_first_high_or_medium_speaks(self):
        self.assertEqual(self.say([hazard("MEDIUM")], 0), "first")

    def test_low_empty_and_busy_are_silent(self):
        self.assertIsNone(self.say([hazard("LOW")], 0))
        self.assertIsNone(self.say([], 0))
        self.assertIsNone(self.say([hazard("HIGH")], 0, speaking=True))

    def test_only_the_primary_hazard_is_spoken(self):
        alert = self.policy.consider([hazard("HIGH", 1, "bicycle"), hazard("HIGH", 2, "car")], now=0)
        self.assertTrue(alert.text.startswith("Warning: bicycle"))

    def test_same_priority_waits_for_the_cooldown(self):
        self.say([hazard("MEDIUM", 1)], 0)
        self.assertIsNone(self.say([hazard("MEDIUM", 2)], 4.9))
        self.assertEqual(self.say([hazard("MEDIUM", 2)], 5.0), "cooldown_elapsed")

    def test_escalation_speaks_immediately(self):
        self.say([hazard("MEDIUM", 1)], 0)
        self.assertEqual(self.say([hazard("HIGH", 1)], 0.2), "escalation")

    def test_lower_priority_waits(self):
        self.say([hazard("HIGH", 1)], 0)
        self.assertIsNone(self.say([hazard("MEDIUM", 2)], 1))
        self.assertEqual(self.say([hazard("MEDIUM", 2)], 3.0), "cooldown_elapsed")

    def test_new_high_object_interrupts_the_cooldown(self):
        self.say([hazard("HIGH", 1)], 0)
        self.assertIsNone(self.say([hazard("HIGH", 1)], 0.5))
        self.assertEqual(self.say([hazard("HIGH", 2, "bicycle")], 0.6), "new_high_object")

    def test_two_high_objects_alternating_do_not_chatter(self):
        self.say([hazard("HIGH", 1)], 0)
        self.say([hazard("HIGH", 2)], 0.1)
        # Both were just announced: neither may interrupt again.
        self.assertIsNone(self.say([hazard("HIGH", 1)], 0.2))
        self.assertIsNone(self.say([hazard("HIGH", 2)], 0.3))

    def test_same_object_is_not_repeated_too_often(self):
        self.say([hazard("MEDIUM", 1)], 0)
        self.assertIsNone(self.say([hazard("MEDIUM", 1)], 6))      # cooldown over, repeat not
        self.assertEqual(self.say([hazard("MEDIUM", 1)], 8.0), "cooldown_elapsed")

    def test_distance_jitter_is_not_a_new_warning(self):
        self.say([hazard("HIGH", 1, distance=2.1)], 0)
        self.assertIsNone(self.say([hazard("HIGH", 1, distance=1.9)], 0.5))

    def test_untracked_objects_follow_the_cooldown(self):
        self.say([hazard("HIGH", None)], 0)
        self.assertIsNone(self.say([hazard("HIGH", None)], 1))
        self.assertEqual(self.say([hazard("HIGH", None)], 3), "cooldown_elapsed")

    def test_reset(self):
        self.say([hazard("HIGH", 1)], 0)
        self.policy.reset()
        self.assertEqual(self.say([hazard("HIGH", 1)], 0.1), "first")

    def test_config_controls_the_rules(self):
        policy = AlertPolicy(config_from_dict({"alerts": {
            "speak_priorities": ["HIGH"], "new_high_object_interrupts": False}}).alerts)
        self.assertIsNone(policy.consider([hazard("MEDIUM")], now=0))
        policy.consider([hazard("HIGH", 1)], now=1)
        self.assertIsNone(policy.consider([hazard("HIGH", 2)], now=1.5))


class TestSentence(unittest.TestCase):

    def test_high_with_direction_distance_and_movement(self):
        text = build_alert_text(hazard("HIGH", name="Bicycle", distance=2.04,
                                       direction="Center", movement="approaching"))
        self.assertEqual(text, "Warning: bicycle ahead, 2 meters, approaching.")

    def test_imminent_and_other_hazards(self):
        text = build_alert_text(hazard("HIGH", distance=6.3, direction="Right",
                                       movement="approaching", imminent=True), other_hazards=2)
        self.assertEqual(text, "Warning: car on your right, 6.5 meters, approaching fast."
                               " 2 more hazards nearby.")

    def test_medium_and_rounding(self):
        self.assertEqual(build_alert_text(hazard("MEDIUM", distance=0.6)),
                         "Caution: car on your left, less than 1 meter.")
        self.assertEqual(build_alert_text(hazard("MEDIUM", distance=1.1)),
                         "Caution: car on your left, 1 meter.")
        self.assertEqual(build_alert_text(hazard("MEDIUM", distance=14.6), 1),
                         "Caution: car on your left, 15 meters. 1 more hazard nearby.")

    def test_unknown_fields(self):
        self.assertEqual(build_alert_text({"priority": "HIGH"}), "Warning: object.")

    def test_smoothed_distance_is_spoken(self):
        text = build_alert_text(hazard("HIGH", distance=9, smoothed_distance_m=3))
        self.assertIn("3 meters", text)

    def test_other_hazards_are_counted_by_the_policy(self):
        alert = AlertPolicy().consider(
            [hazard("HIGH", 1), hazard("MEDIUM", 2), hazard("LOW", 3)], now=0)
        self.assertTrue(alert.text.endswith("1 more hazard nearby."))


if __name__ == "__main__":
    unittest.main()
