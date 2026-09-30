"""Movement and time-to-contact from noisy distances (backend/core/motion.py)."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.motion import MotionEstimator, fit_trend

FOCAL = 700
PERSON_WIDTH = 0.5


def noisy_distance(true_distance, rng, pixel_noise=3):
    """What a box-width estimate reports when the box wobbles by a few pixels."""
    width_px = PERSON_WIDTH * FOCAL / true_distance
    return PERSON_WIDTH * FOCAL / (width_px + rng.uniform(-pixel_noise, pixel_noise))


def run(true_distance_at, fps, seconds, seed=7, pixel_noise=3):
    rng = random.Random(seed)
    motion = MotionEstimator()
    readings = []
    for i in range(int(fps * seconds)):
        t = i / fps
        readings.append(motion.update(1, noisy_distance(true_distance_at(t), rng, pixel_noise), t))
    return readings


class TestNoiseIsNotMotion(unittest.TestCase):
    """The review bug: a still person at 8 m was 'approaching' from jitter."""

    def test_still_objects_never_approach_at_any_frame_rate(self):
        for fps in (3.3, 10, 30):
            for distance in (4, 8, 12):
                for seed in range(5):
                    with self.subTest(fps=fps, distance=distance, seed=seed):
                        readings = run(lambda t: distance, fps, 8, seed)
                        movements = {r.movement for r in readings}
                        self.assertNotIn("approaching", movements)
                        self.assertNotIn("receding", movements)

    def test_smoothing_reduces_jitter(self):
        readings = run(lambda t: 8, 15, 5)
        smoothed = [r.smoothed_distance_m for r in readings[10:]]
        self.assertLess(max(smoothed) - min(smoothed), 1.0)


class TestRealMotion(unittest.TestCase):

    def test_walking_towards_the_camera(self):
        for fps in (3.3, 15, 30):
            with self.subTest(fps=fps):
                readings = run(lambda t: 10 - 1.4 * t, fps, 4)
                self.assertEqual(readings[-1].movement, "approaching")
                self.assertAlmostEqual(readings[-1].closing_speed_mps, 1.4, delta=0.5)
                self.assertIsNotNone(readings[-1].time_to_contact_s)

    def test_walking_away(self):
        readings = run(lambda t: 4 + 1.4 * t, 15, 3)
        self.assertEqual(readings[-1].movement, "receding")
        self.assertIsNone(readings[-1].time_to_contact_s)

    def test_time_to_contact_is_about_distance_over_speed(self):
        readings = run(lambda t: 12 - 4 * t, 30, 2, pixel_noise=0)
        last = readings[-1]
        self.assertAlmostEqual(last.time_to_contact_s,
                               last.smoothed_distance_m / last.closing_speed_mps, delta=0.1)
        self.assertLess(last.time_to_contact_s, 2.5)


class TestFitTrend(unittest.TestCase):

    def test_perfect_line_has_zero_error(self):
        slope, error = fit_trend([(0, 10), (1, 8), (2, 6)])
        self.assertAlmostEqual(slope, -2)
        self.assertAlmostEqual(error, 0)

    def test_scatter_raises_the_error(self):
        _, tight = fit_trend([(0, 10), (1, 8.1), (2, 5.9), (3, 4.0)])
        _, loose = fit_trend([(0, 10), (1, 7.0), (2, 7.0), (3, 4.0)])
        self.assertLess(tight, loose)

    def test_no_time_span(self):
        self.assertIsNone(fit_trend([(1, 5), (1, 6)]))


class TestNotEnoughInformation(unittest.TestCase):

    def test_first_frames_have_no_movement(self):
        motion = MotionEstimator()
        self.assertIsNone(motion.update(1, 5, 0.0).movement)
        self.assertIsNone(motion.update(1, 4, 0.1).movement)      # too few samples
        self.assertIsNone(motion.update(1, 3, 0.2).movement)      # span < min_span_s

    def test_missing_distance_or_track_is_harmless(self):
        motion = MotionEstimator()
        for track_id, distance in [(None, 5), (1, None), (1, 0), (1, "far"), (1, float("nan"))]:
            reading = motion.update(track_id, distance, 0.0)
            self.assertIsNone(reading.movement)
            self.assertIsNone(reading.smoothed_distance_m)

    def test_forget_except_drops_other_tracks(self):
        motion = MotionEstimator()
        motion.update(1, 5, 0)
        motion.update(2, 5, 0)
        motion.forget_except({2})
        # Track 1 starts over: its first smoothed value is the raw value again.
        self.assertEqual(motion.update(1, 9, 0.1).smoothed_distance_m, 9)


if __name__ == "__main__":
    unittest.main()
