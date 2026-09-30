"""End-to-end prioritization over several frames (backend/core/pipeline.py)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hazard_config import config_from_dict
from core.pipeline import HazardPrioritizer


def frame(*items):
    """items: (name, distance, direction, x-offset of a 100x100 box)."""
    return [{"label": name, "distance": distance, "direction": direction,
             "confidence": 0.9, "bbox": [x, 100, x + 100, 200]}
            for name, distance, direction, x in items]


class TestPipeline(unittest.TestCase):

    def test_task_scene_through_the_pipeline(self):
        ranked = HazardPrioritizer().process(frame(
            ("Person", 8, None, 0), ("Car", 5, None, 300), ("Bicycle", 2, None, 600)), now=0)
        self.assertEqual([d["label"] for d in ranked], ["Bicycle", "Car", "Person"])
        self.assertEqual([d["priority"] for d in ranked], ["HIGH", "MEDIUM", "LOW"])
        self.assertTrue(all(d["track_id"] is not None for d in ranked))

    def test_empty_frames(self):
        pipeline = HazardPrioritizer()
        self.assertEqual(pipeline.process([], now=0), [])
        self.assertEqual(pipeline.process(None, now=1), [])

    def test_detections_without_bbox_still_score(self):
        ranked = HazardPrioritizer().process(
            [{"object": "car", "distance_m": 5, "direction": "Left"}], now=0)
        self.assertEqual(ranked[0]["risk_score"], 60.0)
        self.assertIsNone(ranked[0]["track_id"])

    def test_approaching_object_gains_movement_and_escalates(self):
        pipeline = HazardPrioritizer()
        last = None
        for i in range(15):                     # 1.5 s at 10 fps, 12 m -> 3 m
            t = i / 10
            ranked = pipeline.process(frame(("Car", 12 - 6 * t, "Left", 100)), now=t)
            last = ranked[0]
        self.assertEqual(last["movement"], "approaching")
        self.assertIn("time_to_contact_s", last)
        self.assertEqual(last["priority"], "HIGH")

    def test_caller_supplied_movement_is_kept(self):
        pipeline = HazardPrioritizer()
        for i in range(10):
            ranked = pipeline.process([{"object": "car", "distance_m": 5, "direction": "Left",
                                        "movement": "receding", "bbox": [0, 0, 50, 50]}],
                                      now=i / 10)
        self.assertEqual(ranked[0]["movement"], "receding")

    def test_reset_forgets_history(self):
        pipeline = HazardPrioritizer()
        first = pipeline.process(frame(("Car", 5, "Left", 0)), now=0)[0]["track_id"]
        pipeline.reset()
        again = pipeline.process(frame(("Car", 5, "Left", 0)), now=0.1)[0]["track_id"]
        self.assertEqual(first, again)          # ids restart at 1


class TestPriorityHold(unittest.TestCase):
    """A score wobbling around a cutoff must not make the warning flicker."""

    def setUp(self):
        self.pipeline = HazardPrioritizer(config_from_dict({
            "motion": {"min_speed_mps": 100}}))  # keep movement out of this test

    def _priority(self, level, t):
        # A person 2 m away scores 70 (HIGH) in the centre, 64 (MEDIUM) on the side.
        direction = "Center" if level == "high" else "Left"
        return self.pipeline.process(frame(("Person", 2, direction, 100)), now=t)[0]

    def test_rises_immediately(self):
        self.assertEqual(self._priority("medium", 0.0)["priority"], "MEDIUM")
        self.assertEqual(self._priority("high", 0.1)["priority"], "HIGH")

    def test_falls_only_after_the_hold(self):
        self._priority("high", 0.0)
        held = self._priority("medium", 0.3)
        self.assertEqual(held["priority"], "HIGH")
        self.assertEqual(held["raw_priority"], "MEDIUM")
        self.assertTrue(held["priority_held"])
        self.assertEqual(held["risk_score"], 64.0)      # the score itself is honest
        self.assertIn("Held at HIGH", held["reason"])

        released = self._priority("medium", 1.2)
        self.assertEqual(released["priority"], "MEDIUM")
        self.assertFalse(released["priority_held"])

    def test_flicker_every_frame_stays_high(self):
        priorities = [self._priority("high" if i % 2 == 0 else "medium", i / 10)["priority"]
                      for i in range(20)]
        self.assertEqual(set(priorities), {"HIGH"})

    def test_untracked_objects_are_never_held(self):
        pipeline = HazardPrioritizer()
        pipeline.process([{"object": "person", "distance_m": 2, "direction": "Center"}], now=0)
        after = pipeline.process([{"object": "person", "distance_m": 2, "direction": "Left"}], now=0.1)
        self.assertEqual(after[0]["priority"], "MEDIUM")


if __name__ == "__main__":
    unittest.main()
