"""
Scoring, classification and ranking (backend/core/risk.py).

Run all hazard tests from the repository root:
    python -m unittest discover -s backend/tests -v
"""

import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hazard_config import DEFAULT_CONFIG, config_from_dict
from core.risk import (classify, rank_detections, score_detections, score_single,
                       sort_and_rank, type_group_for)


# ============================ 1. Worked examples ============================

class TestWorkedExamples(unittest.TestCase):
    """Every row of the worked-example table in docs/HAZARD_SCORING.md."""

    CASES = [
        # name, distance, direction, movement, score, priority
        ("person", 8, "Left", None, 32.0, "LOW"),
        ("person", 8, None, None, 35.0, "LOW"),
        ("person", 8, "Center", None, 38.0, "LOW"),
        ("car", 5, "Left", None, 60.0, "MEDIUM"),
        ("car", 5, None, None, 63.0, "MEDIUM"),
        ("car", 5, "Center", None, 66.0, "MEDIUM"),
        ("bicycle", 2, "Left", None, 74.0, "HIGH"),
        ("bicycle", 2, None, None, 77.0, "HIGH"),
        ("bicycle", 2, "Center", None, 80.0, "HIGH"),
        ("bottle", 2, "Center", None, 21.0, "LOW"),
        ("bench", 2, "Center", None, 42.2, "MEDIUM"),
        ("car", 5, "Left", "approaching", 72.0, "HIGH"),
        ("car", 5, "Left", "receding", 52.0, "MEDIUM"),
        ("person", 2, "Center", None, 70.0, "HIGH"),
        ("person", 2, "Left", None, 64.0, "MEDIUM"),
        ("car", 12, "Left", None, 40.0, "MEDIUM"),
        ("car", 13, "Left", None, 36.0, "LOW"),
        ("car", None, "Left", None, 54.0, "MEDIUM"),
        ("dog", 2, "Center", None, 57.2, "MEDIUM"),
        ("traffic light", 5, "Center", None, 32.3, "LOW"),
        ("", 2, "Center", None, 50.8, "MEDIUM"),
    ]

    def test_table(self):
        for name, distance, direction, movement, score, priority in self.CASES:
            with self.subTest(name=name, distance=distance, direction=direction,
                              movement=movement):
                result = score_single(name, distance, direction, movement)
                self.assertEqual(result["risk_score"], score)
                self.assertEqual(result["priority"], priority)


class TestTaskExampleScene(unittest.TestCase):
    """The scene from the task: Pedestrian 8 m, Car 5 m, Bike 2 m."""

    def test_bike_is_primary_in_every_position(self):
        for direction in ("Left", "Center", "Right", None):
            with self.subTest(direction=direction):
                ranked = rank_detections([
                    {"object": "person", "distance_m": 8, "direction": direction},
                    {"object": "car", "distance_m": 5, "direction": direction},
                    {"object": "bicycle", "distance_m": 2, "direction": direction},
                ])
                self.assertEqual([d["object"] for d in ranked], ["bicycle", "car", "person"])
                self.assertEqual([d["priority"] for d in ranked], ["HIGH", "MEDIUM", "LOW"])
                self.assertEqual([d["rank"] for d in ranked], [1, 2, 3])

    def test_scores_with_position_omitted(self):
        ranked = rank_detections([
            {"object": "person", "distance_m": 8},
            {"object": "car", "distance_m": 5},
            {"object": "bicycle", "distance_m": 2},
        ])
        self.assertEqual([d["risk_score"] for d in ranked], [77.0, 63.0, 35.0])


# ============================ 2. Explainability =============================

class TestBreakdownAndReason(unittest.TestCase):

    def test_breakdown_adds_up_to_the_score(self):
        for args in [("bicycle", 2, "Center", "approaching"), ("car", 5, "Left", None),
                     ("bench", 2, "Center", None), ("dog", None, None, "receding")]:
            with self.subTest(args=args):
                result = score_single(*args, confidence=0.9)
                self.assertAlmostEqual(sum(result["breakdown"].values()),
                                       result["risk_score"], places=1)

    def test_breakdown_values(self):
        result = score_single("dog", 2, "Center", "approaching", confidence=0.2)
        self.assertEqual(result["breakdown"], {
            "type": 16.0, "distance": 35.2, "position": 6.0,
            "movement": 12.0, "confidence": -15.0,
        })
        self.assertEqual(result["category"], "animal")

    def test_reason_names_the_facts(self):
        reason = score_single("bicycle", 2, "Center", "approaching")["reason"]
        self.assertTrue(reason.startswith("Bicycle, 2 m, center, approaching."))
        for words in ("it is a bicycle", "very close", "walking line", "getting closer"):
            self.assertIn(words, reason.lower())

    def test_reason_for_unknowns(self):
        reason = score_single(None, None, None, None)["reason"]
        self.assertIn("Unknown object", reason)
        self.assertIn("distance unknown", reason)
        self.assertIn("position unknown", reason)

    def test_reason_mentions_low_confidence(self):
        reason = score_single("car", 5, "Left", confidence=0.25)["reason"]
        self.assertIn("unsure (25% confidence)", reason)


# ========================= 3. Confidence & imminent =========================

class TestConfidence(unittest.TestCase):

    def test_low_confidence_costs_points(self):
        sure = score_single("bicycle", 2, "Center", confidence=0.9)
        unsure = score_single("bicycle", 2, "Center", confidence=0.3)
        self.assertEqual(sure["risk_score"] - unsure["risk_score"], 15.0)
        self.assertEqual(unsure["priority"], "MEDIUM")

    def test_missing_or_invalid_confidence_is_neutral(self):
        base = score_single("car", 5, "Left")["risk_score"]
        for confidence in (None, "high", 1.5, -0.2, float("nan"), True):
            with self.subTest(confidence=confidence):
                self.assertEqual(score_single("car", 5, "Left", confidence=confidence)["risk_score"], base)

    def test_cutoff_is_inclusive_of_threshold(self):
        at = score_single("car", 5, "Left", confidence=0.40)
        below = score_single("car", 5, "Left", confidence=0.39)
        self.assertEqual(at["breakdown"]["confidence"], 0.0)
        self.assertEqual(below["breakdown"]["confidence"], -15.0)


class TestImminent(unittest.TestCase):

    def test_fast_approach_is_high_even_when_far(self):
        result = score_single("car", 10, "Left", "approaching", time_to_contact_s=1.5)
        self.assertEqual(result["risk_score"], 52.0)       # MEDIUM by points alone
        self.assertEqual(result["priority"], "HIGH")
        self.assertTrue(result["imminent"])
        self.assertIn("Raised to HIGH", result["reason"])

    def test_slow_approach_is_not_imminent(self):
        result = score_single("car", 10, "Left", "approaching", time_to_contact_s=6)
        self.assertEqual(result["priority"], "MEDIUM")
        self.assertFalse(result["imminent"])

    def test_ttc_without_approaching_is_ignored(self):
        for movement in (None, "stationary", "receding"):
            result = score_single("car", 10, "Left", movement, time_to_contact_s=1)
            self.assertFalse(result["imminent"])

    def test_low_confidence_cannot_trigger_imminent(self):
        result = score_single("car", 10, "Left", "approaching",
                              time_to_contact_s=1, confidence=0.2)
        self.assertFalse(result["imminent"])

    def test_rule_can_be_switched_off(self):
        cfg = config_from_dict({"scoring": {"imminent_ttc_s": None}})
        result = score_single("car", 10, "Left", "approaching",
                              time_to_contact_s=1, config=cfg.scoring)
        self.assertEqual(result["priority"], "MEDIUM")


# ============================ 4. Robust inputs ==============================

class TestRobustInputs(unittest.TestCase):

    def test_case_and_whitespace_do_not_matter(self):
        base = score_single("bicycle", 2, "Center", "approaching")["risk_score"]
        for args in [("  BICYCLE ", 2, "center", "Approaching"),
                     ("Bicycle", 2, " CENTER ", " APPROACHING ")]:
            self.assertEqual(score_single(*args)["risk_score"], base)

    def test_movement_synonyms(self):
        for word, same_as in [("closing", "approaching"), ("moving away", "receding"),
                              ("still", "stationary"), ("sideways", None)]:
            with self.subTest(word=word):
                self.assertEqual(score_single("car", 5, "Left", word)["risk_score"],
                                 score_single("car", 5, "Left", same_as)["risk_score"])

    def test_bad_distances_count_as_unknown(self):
        for distance in (None, 0, -5, "far away", float("nan"), float("inf"),
                         True, False, [1, 2], {"m": 5}):
            with self.subTest(distance=distance):
                result = score_single("car", distance, "Left")
                self.assertEqual(result["risk_score"], 54.0)
                self.assertIn("distance unknown", result["reason"])

    def test_numeric_string_distance_is_accepted(self):
        self.assertEqual(score_single("car", "5", "Left")["risk_score"], 60.0)

    def test_bad_positions_count_as_unknown(self):
        for direction in (None, "", "ahead", 3, ["Left"]):
            with self.subTest(direction=direction):
                self.assertEqual(score_single("car", 5, direction)["risk_score"], 63.0)

    def test_non_string_names_do_not_crash(self):
        for name in (5, 3.2, ["car"], {"n": 1}, b"car"):
            with self.subTest(name=name):
                result = score_single(name, 2, "Center")
                self.assertEqual(result["risk_score"], 50.8)   # blank-name row
                self.assertIn("Unknown object", result["reason"])

    def test_blank_and_other_names(self):
        self.assertEqual(type_group_for("   ")[0], "unknown")
        self.assertEqual(type_group_for("toaster")[0], "other")
        self.assertEqual(score_single("toaster", 2, "Center")["risk_score"], 21.0)

    def test_empty_and_all_none_detections(self):
        self.assertEqual(rank_detections([]), [])
        self.assertEqual(rank_detections(None), [])
        ranked = rank_detections([{"object": None, "distance_m": None, "direction": None}])
        self.assertEqual(ranked[0]["rank"], 1)


# ======================== 5. Consistency guarantees =========================
# Checked across every class group, distance band, position and movement.

_NAMES = ["car", "bicycle", "person", "skateboard", "dog", "bench", "bottle", ""]
_DISTANCES = [0.5, 1, 2, 2.01, 3, 5, 5.01, 7, 8, 8.01, 10, 12, 12.01, 20, 50]
_POSITIONS = ["Left", "Center", "Right", None]
_MOVEMENTS = ["approaching", "stationary", "receding", None]


def _all_inputs():
    return itertools.product(_NAMES, _DISTANCES, _POSITIONS, _MOVEMENTS)


class TestConsistency(unittest.TestCase):

    def test_score_is_always_between_0_and_100(self):
        for args in _all_inputs():
            score = score_single(*args, confidence=0.1)["risk_score"]
            self.assertTrue(0 <= score <= 100, args)

    def test_moving_closer_never_lowers_the_score(self):
        for name, position, movement in itertools.product(_NAMES, _POSITIONS, _MOVEMENTS):
            scores = [score_single(name, d, position, movement)["risk_score"]
                      for d in _DISTANCES]
            self.assertEqual(scores, sorted(scores, reverse=True),
                             (name, position, movement))

    def test_approaching_never_lowers_the_score(self):
        for name, distance, position in itertools.product(_NAMES, _DISTANCES, _POSITIONS):
            s = {m: score_single(name, distance, position, m)["risk_score"] for m in _MOVEMENTS}
            self.assertGreaterEqual(s["approaching"], s["stationary"])
            self.assertEqual(s["stationary"], s[None])
            self.assertGreaterEqual(s["stationary"], s["receding"])

    def test_walking_line_never_lowers_the_score(self):
        for name, distance, movement in itertools.product(_NAMES, _DISTANCES, _MOVEMENTS):
            s = {p: score_single(name, distance, p, movement)["risk_score"] for p in _POSITIONS}
            self.assertEqual(s["Left"], s["Right"])
            self.assertGreaterEqual(s["Center"], s[None])
            self.assertGreaterEqual(s[None], s["Left"])

    def test_priority_follows_score(self):
        for args in _all_inputs():
            result = score_single(*args)
            self.assertEqual(result["priority"], classify(result["risk_score"]), args)

    def test_same_input_same_output(self):
        for args in _all_inputs():
            self.assertEqual(score_single(*args), score_single(*args))

    def test_ranking_ignores_input_order(self):
        scene = [
            {"object": "person", "distance_m": 8, "direction": "Left"},
            {"object": "car", "distance_m": 5, "direction": "Right"},
            {"object": "bicycle", "distance_m": 2, "direction": "Center"},
            {"object": "dog", "distance_m": 3, "direction": None},
            {"object": "bottle", "distance_m": 1, "direction": "Center"},
        ]
        expected = [d["object"] for d in rank_detections(scene)]
        for order in itertools.permutations(scene):
            self.assertEqual([d["object"] for d in rank_detections(list(order))], expected)


# ============================ 6. Tie-breaking ===============================

def _scored(name, distance, direction, score=60.0, priority="MEDIUM"):
    return {"object": name, "distance_m": distance, "direction": direction,
            "risk_score": score, "priority": priority}


class TestSortOrder(unittest.TestCase):
    """sort_and_rank takes pre-scored items, so every tie rule is testable."""

    def _names(self, items):
        return [d["object"] for d in sort_and_rank(items)]

    def test_priority_beats_score(self):
        # A held HIGH (score 68) stays above a MEDIUM that scores 69.
        items = [_scored("car", 5, "Left", 69.0, "MEDIUM"),
                 _scored("bicycle", 3, "Left", 68.0, "HIGH")]
        self.assertEqual(self._names(items), ["bicycle", "car"])

    def test_higher_score_first(self):
        items = [_scored("a", 5, "Left", 50.0), _scored("b", 5, "Left", 60.0)]
        self.assertEqual(self._names(items), ["b", "a"])

    def test_nearer_first_and_unknown_distance_last(self):
        items = [_scored("far", 9, "Left"), _scored("unknown", None, "Left"),
                 _scored("near", 3, "Left")]
        self.assertEqual(self._names(items), ["near", "far", "unknown"])

    def test_center_then_side_then_unknown_position(self):
        items = [_scored("unknown", 5, None), _scored("side", 5, "Right"),
                 _scored("center", 5, "Center")]
        self.assertEqual(self._names(items), ["center", "side", "unknown"])

    def test_name_alphabetical_then_detector_order(self):
        items = [_scored("car", 5, "Left"), _scored("Bus", 5, "Left"),
                 {**_scored("car", 5, "Right"), "id": "second car"}]
        ranked = sort_and_rank(items)
        self.assertEqual([d["object"] for d in ranked], ["Bus", "car", "car"])
        self.assertEqual(ranked[2].get("id"), "second car")

    def test_ranks_start_at_one(self):
        ranked = sort_and_rank([_scored("a", 1, "Left"), _scored("b", 2, "Left")])
        self.assertEqual([d["rank"] for d in ranked], [1, 2])


# =========================== 7. Dict handling ===============================

class TestDetectionDicts(unittest.TestCase):

    def test_both_shapes_score_the_same(self):
        api = rank_detections([{"object": "car", "distance_m": 5, "direction": "Left"}])[0]
        live = rank_detections([{"label": "Car", "distance": 5, "direction": "Left"}])[0]
        self.assertEqual(api["risk_score"], live["risk_score"])

    def test_keys_are_kept_and_added(self):
        det = {"object": "car", "direction": "Left", "distance_m": 5.0,
               "bbox": [100, 200, 300, 400]}
        item = rank_detections([det])[0]
        for key, value in det.items():
            self.assertEqual(item[key], value)
        for key in ("risk_score", "priority", "reason", "breakdown", "category",
                    "imminent", "rank"):
            self.assertIn(key, item)

    def test_inputs_are_not_mutated(self):
        det = {"object": "car", "distance_m": 5, "direction": "Left"}
        rank_detections([det])
        score_detections([det])
        self.assertEqual(det, {"object": "car", "distance_m": 5, "direction": "Left"})

    def test_smoothed_distance_is_preferred(self):
        det = {"object": "car", "distance_m": 9, "smoothed_distance_m": 4, "direction": "Left"}
        self.assertEqual(rank_detections([det])[0]["breakdown"]["distance"], 26.0)

    def test_confidence_and_ttc_are_read_from_dicts(self):
        det = {"object": "car", "distance_m": 10, "direction": "Left",
               "movement": "approaching", "time_to_contact_s": 1.0, "confidence": 0.9}
        self.assertTrue(rank_detections([det])[0]["imminent"])


# ============================ 8. Configurability ============================

class TestConfigDrivesScoring(unittest.TestCase):

    def test_changed_cutoff_changes_priority(self):
        cfg = config_from_dict({"scoring": {"high_cutoff": 85}})
        self.assertEqual(score_single("bicycle", 2, "Center", config=cfg.scoring)["priority"], "MEDIUM")
        self.assertEqual(score_single("bicycle", 2, "Center")["priority"], "HIGH")

    def test_changed_weight_changes_score(self):
        cfg = config_from_dict({"scoring": {"type_groups": {"person": {"weight": 30}}}})
        self.assertEqual(score_single("person", 8, "Left", config=cfg.scoring)["risk_score"], 42.0)

    def test_new_bands_change_reason_words(self):
        cfg = config_from_dict({"scoring": {"distance_bands": [
            [3, 40, "right beside you"], [None, 5, "somewhere out there"]]}})
        reason = score_single("car", 2.5, "Left", config=cfg.scoring)["reason"]
        self.assertIn("right beside you", reason)

    def test_default_config_is_used_when_none_given(self):
        self.assertEqual(score_single("car", 5, "Left", config=None),
                         score_single("car", 5, "Left", config=DEFAULT_CONFIG.scoring))


if __name__ == "__main__":
    unittest.main()
