"""Stable track ids across frames (backend/core/tracking.py)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tracking import IouTracker, iou


def det(name, box):
    return {"object": name, "bbox": list(box)}


class TestIou(unittest.TestCase):

    def test_values(self):
        self.assertEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)


class TestIouTracker(unittest.TestCase):

    def test_same_object_keeps_its_id(self):
        tracker = IouTracker()
        first = tracker.update([det("car", (100, 100, 200, 180))], now=0.0)
        second = tracker.update([det("car", (104, 102, 206, 184))], now=0.1)
        self.assertEqual(first[0]["track_id"], second[0]["track_id"])

    def test_two_objects_are_not_swapped(self):
        tracker = IouTracker()
        a, b = tracker.update([det("car", (0, 0, 100, 80)), det("car", (300, 0, 400, 80))], now=0)
        # Next frame lists them in the other order.
        b2, a2 = tracker.update([det("car", (305, 0, 405, 80)), det("car", (3, 0, 103, 80))], now=0.1)
        self.assertEqual(a["track_id"], a2["track_id"])
        self.assertEqual(b["track_id"], b2["track_id"])

    def test_class_change_is_a_new_object(self):
        tracker = IouTracker()
        car = tracker.update([det("car", (0, 0, 100, 80))], now=0)[0]
        bus = tracker.update([det("bus", (0, 0, 100, 80))], now=0.1)[0]
        self.assertNotEqual(car["track_id"], bus["track_id"])

    def test_both_detection_shapes_and_case(self):
        tracker = IouTracker()
        a = tracker.update([{"object": "car", "bbox": [0, 0, 100, 80]}], now=0)[0]
        b = tracker.update([{"label": "Car", "bbox": [2, 0, 102, 80]}], now=0.1)[0]
        self.assertEqual(a["track_id"], b["track_id"])

    def test_non_overlapping_box_is_a_new_object(self):
        tracker = IouTracker()
        a = tracker.update([det("person", (0, 0, 50, 150))], now=0)[0]
        b = tracker.update([det("person", (600, 0, 650, 150))], now=0.1)[0]
        self.assertNotEqual(a["track_id"], b["track_id"])

    def test_short_gap_is_bridged_and_long_gap_forgets(self):
        tracker = IouTracker()
        a = tracker.update([det("car", (0, 0, 100, 80))], now=0.0)[0]
        tracker.update([], now=0.5)
        b = tracker.update([det("car", (0, 0, 100, 80))], now=0.9)[0]
        self.assertEqual(a["track_id"], b["track_id"])
        tracker.update([], now=2.5)
        c = tracker.update([det("car", (0, 0, 100, 80))], now=2.6)[0]
        self.assertNotEqual(b["track_id"], c["track_id"])

    def test_missing_or_bad_bbox_gets_none(self):
        tracker = IouTracker()
        out = tracker.update([{"object": "car"}, det("car", (5, 5, 5, 20)),
                              {"object": "car", "bbox": "oops"}], now=0)
        self.assertEqual([d["track_id"] for d in out], [None, None, None])

    def test_inputs_are_not_mutated_and_reset_restarts_ids(self):
        tracker = IouTracker()
        original = det("car", (0, 0, 10, 10))
        tracker.update([original], now=0)
        self.assertNotIn("track_id", original)
        tracker.reset()
        self.assertEqual(tracker.active_ids, set())
        self.assertEqual(tracker.update([original], now=1)[0]["track_id"], 1)


if __name__ == "__main__":
    unittest.main()
