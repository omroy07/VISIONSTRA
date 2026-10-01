"""Monocular distance estimation (backend/core/distance.py)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.distance import (REFERENCE_SIZES_M, calibrate_focal_length,
                           estimate_distance, estimate_distance_from_box)

FOCAL = 700
W, H = 1280, 720


def box_for(label, distance, x=400, y=200):
    """The box the given object would produce at ``distance`` metres."""
    real_w, real_h = REFERENCE_SIZES_M[label]
    width = real_w * FOCAL / distance
    height = real_h * FOCAL / distance
    return (x, y, x + width, y + height)


class TestEstimateFromBox(unittest.TestCase):

    def test_recovers_true_distance_for_each_class(self):
        for label in ("person", "car", "bus", "bicycle", "dog"):
            for distance in (3, 6, 10):
                with self.subTest(label=label, distance=distance):
                    box = box_for(label, distance)
                    self.assertAlmostEqual(
                        estimate_distance_from_box(box, label, 4000, 4000, FOCAL), distance, places=1)

    def test_car_is_no_longer_estimated_3x_too_close(self):
        box = box_for("car", 10)
        old_style = estimate_distance(box[2] - box[0], 0.5, FOCAL)
        new_style = estimate_distance_from_box(box, "Car", 4000, 4000, FOCAL)
        self.assertLess(old_style, 3.0)
        self.assertAlmostEqual(new_style, 10.0, places=1)

    def test_height_is_used_when_the_object_turns(self):
        # A car seen side-on is ~4.5 m long: width lies, height does not.
        box = (100, 100, 100 + 4.5 * FOCAL / 8, 100 + 1.5 * FOCAL / 8)
        self.assertAlmostEqual(estimate_distance_from_box(box, "car", W, H, FOCAL), 8.0, places=1)

    def test_width_is_used_when_top_or_bottom_is_cut_off(self):
        # Person close to the camera: feet are below the frame.
        width = 0.5 * FOCAL / 1.5
        box = (500, 50, 500 + width, H)
        self.assertAlmostEqual(estimate_distance_from_box(box, "person", W, H, FOCAL), 1.5, places=1)

    def test_nearer_estimate_when_cut_off_on_both_axes(self):
        box = (0, 0, W, H)
        distance = estimate_distance_from_box(box, "bus", W, H, FOCAL)
        self.assertEqual(distance, round(min(2.55 * FOCAL / W, 3.2 * FOCAL / H), 2))

    def test_unknown_class_falls_back_to_half_metre_width(self):
        box = (100, 100, 170, 400)
        self.assertEqual(estimate_distance_from_box(box, "suitcase", W, H, FOCAL), 5.0)
        self.assertEqual(estimate_distance_from_box(box, None, W, H, FOCAL), 5.0)

    def test_invalid_boxes_give_none(self):
        self.assertIsNone(estimate_distance_from_box((10, 10, 10, 50), "car", W, H, FOCAL))
        self.assertIsNone(estimate_distance_from_box((10, 50, 40, 20), "car", W, H, FOCAL))
        self.assertIsNone(estimate_distance(0))
        self.assertIsNone(estimate_distance(None))


class TestCalibration(unittest.TestCase):

    def test_round_trip(self):
        focal = calibrate_focal_length(pixel_size=175, distance_m=2, real_size_m=0.5)
        self.assertEqual(focal, 700)
        self.assertAlmostEqual(estimate_distance(175, 0.5, focal), 2.0)

    def test_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            calibrate_focal_length(0, 2, 0.5)


if __name__ == "__main__":
    unittest.main()
