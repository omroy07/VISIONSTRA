"""Loading, overriding and validating the hazard config."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hazard_config import (CONFIG_ENV_VAR, DEFAULT_CONFIG, HazardConfig,
                                config_from_dict, load_config)


class TestDefaults(unittest.TestCase):

    def test_default_config_is_valid(self):
        self.assertIs(HazardConfig().validate().__class__, HazardConfig)

    def test_no_path_and_no_env_gives_defaults(self):
        self.assertIs(load_config(environ={}), DEFAULT_CONFIG)

    def test_to_dict_is_json_serialisable(self):
        json.dumps(DEFAULT_CONFIG.to_dict())

    def test_example_file_loads_and_matches_defaults(self):
        example = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "hazard_config.example.json")
        self.assertEqual(load_config(example), DEFAULT_CONFIG)


class TestOverrides(unittest.TestCase):

    def test_partial_override_keeps_everything_else(self):
        cfg = config_from_dict({"scoring": {"high_cutoff": 75}})
        self.assertEqual(cfg.scoring.high_cutoff, 75)
        self.assertEqual(cfg.scoring.medium_cutoff, DEFAULT_CONFIG.scoring.medium_cutoff)
        self.assertEqual(cfg.alerts, DEFAULT_CONFIG.alerts)

    def test_type_group_merge_and_new_group(self):
        cfg = config_from_dict({"scoring": {"type_groups": {
            "vehicle": {"weight": 40},
            "luggage": {"weight": 10, "distance_factor": 0.5,
                        "description": "luggage", "classes": ["Suitcase"]},
        }}})
        self.assertEqual(cfg.scoring.type_groups["vehicle"].weight, 40)
        self.assertIn("car", cfg.scoring.type_groups["vehicle"].classes)
        self.assertEqual(cfg.scoring.type_groups["luggage"].classes, ("suitcase",))

    def test_dict_values_merge(self):
        cfg = config_from_dict({"alerts": {"cooldown_s": {"high": 1.5}}})
        self.assertEqual(cfg.alerts.cooldown_s, {"HIGH": 1.5, "MEDIUM": 5.0})

    def test_defaults_are_not_modified_by_overrides(self):
        config_from_dict({"scoring": {"position_points": {"center": 20}}})
        self.assertEqual(DEFAULT_CONFIG.scoring.position_points["center"], 6)

    def test_load_from_file_and_env(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"camera": {"focal_length_px": 640}}, handle)
        try:
            self.assertEqual(load_config(handle.name).camera.focal_length_px, 640)
            from_env = load_config(environ={CONFIG_ENV_VAR: handle.name})
            self.assertEqual(from_env.camera.focal_length_px, 640)
        finally:
            os.unlink(handle.name)


class TestValidation(unittest.TestCase):

    BAD = [
        ({"scoring": {"no_such_setting": 1}}, "Unknown setting scoring.no_such_setting"),
        ({"nonsense": {}}, "Unknown setting nonsense"),
        ({"scoring": {"medium_cutoff": 80}}, "medium_cutoff must be below"),
        ({"scoring": {"distance_bands": [[2, 10, "a"], [None, 20, "b"]]}}, "must not increase"),
        ({"scoring": {"distance_bands": [[5, 40, "a"], [2, 20, "b"], [None, 1, "c"]]}},
         "positive and increasing"),
        ({"scoring": {"distance_bands": [[5, 40, "a"]]}}, "must end with an open row"),
        ({"scoring": {"movement_points": {"approaching": -5}}}, "approaching >= stationary"),
        ({"scoring": {"type_groups": {"person": {"classes": ["person", "car"]}}}},
         "class 'car' is in both"),
        ({"scoring": {"type_groups": {"new": {"weight": 3}}}}, "needs"),
        ({"scoring": {"low_confidence_penalty": 5}}, "zero or negative"),
        ({"motion": {"smoothing_alpha": 0}}, "smoothing_alpha"),
        ({"tracking": {"iou_threshold": 1.5}}, "iou_threshold"),
        ({"alerts": {"speak_priorities": ["URGENT"]}}, "unknown priority"),
        ({"camera": {"focal_length_px": 0}}, "focal_length_px"),
    ]

    def test_bad_values_are_rejected_with_a_clear_message(self):
        for overrides, message in self.BAD:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError) as caught:
                    config_from_dict(overrides)
                self.assertIn(message, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
