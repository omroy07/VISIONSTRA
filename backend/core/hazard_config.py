"""
Every tunable number used by hazard prioritization lives in this file.

To change behaviour, either edit a default below or point the environment
variable ``VISIONSTRA_HAZARD_CONFIG`` at a JSON file that holds only the
values you want to override. Anything not listed keeps its default:

    {
      "scoring": {"high_cutoff": 75, "type_groups": {"vehicle": {"weight": 38}}},
      "alerts":  {"cooldown_s": {"HIGH": 2.0}},
      "camera":  {"focal_length_px": 640}
    }

Unknown keys and inconsistent values (for example a MEDIUM cutoff above the
HIGH cutoff) raise ``ValueError`` at start-up instead of misbehaving later.
docs/HAZARD_SCORING.md explains what each value does.
"""

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace

PRIORITY_LEVELS = ("LOW", "MEDIUM", "HIGH")
POSITIONS = ("left", "center", "right")
MOVEMENTS = ("approaching", "stationary", "receding")

CONFIG_ENV_VAR = "VISIONSTRA_HAZARD_CONFIG"


# ================================ SCORING ===================================

@dataclass(frozen=True)
class TypeGroup:
    """How much one family of object classes matters."""
    weight: float               # points added just for being this kind of object
    distance_factor: float      # how strongly closeness matters for this kind
    description: str            # used in the reason text: "it is <description>"
    classes: tuple = ()         # lower-case YOLO/COCO class names


def _default_type_groups():
    return {
        "vehicle": TypeGroup(
            34, 1.0, "a vehicle",
            ("car", "bus", "truck", "train", "motorcycle"),
        ),
        "bicycle": TypeGroup(30, 1.0, "a bicycle", ("bicycle",)),
        "person": TypeGroup(20, 1.0, "a person", ("person",)),
        "rider_equipment": TypeGroup(
            18, 0.9, "rider equipment",
            ("skateboard", "skis", "snowboard", "surfboard"),
        ),
        "animal": TypeGroup(
            16, 0.8, "an animal",
            ("dog", "cat", "horse", "cow", "sheep",
             "bird", "bear", "elephant", "zebra", "giraffe"),
        ),
        "street_obstacle": TypeGroup(
            12, 0.55, "a street obstacle",
            ("bench", "chair", "couch", "potted plant", "fire hydrant",
             "parking meter", "traffic light", "stop sign", "dining table"),
        ),
    }


@dataclass(frozen=True)
class ScoringConfig:
    type_groups: dict = field(default_factory=_default_type_groups)
    # Any non-blank class name that is not in a group above (bottle, cup, ...).
    other_group: TypeGroup = TypeGroup(4, 0.25, "a low-risk object")
    # Blank or missing class name: the detector saw *something*.
    unknown_group: TypeGroup = TypeGroup(14, 0.7, "an unidentified object")

    # (upper bound in metres, inclusive; points; words for the reason text).
    # Checked in order. The last row must have upper bound None ("beyond").
    distance_bands: tuple = (
        (2.0, 44, "very close"),
        (5.0, 26, "nearby"),
        (8.0, 12, "at a moderate distance"),
        (12.0, 6, "far"),
        (None, 2, "very far"),
    )
    distance_missing_points: float = 20

    position_points: dict = field(
        default_factory=lambda: {"center": 6, "left": 0, "right": 0})
    position_missing_points: float = 3

    movement_points: dict = field(
        default_factory=lambda: {"approaching": 12, "stationary": 0, "receding": -8})
    movement_synonyms: dict = field(
        default_factory=lambda: {"closing": "approaching",
                                 "moving away": "receding",
                                 "still": "stationary"})
    movement_missing_points: float = 0

    # Detections the model is unsure about lose points instead of vanishing.
    low_confidence_below: float = 0.40
    low_confidence_penalty: float = -15

    # An approaching object this many seconds (or fewer) from reaching the
    # camera is HIGH whatever its score. None switches the rule off.
    imminent_ttc_s: float = 2.5

    high_cutoff: float = 70
    medium_cutoff: float = 40


# ============================ MOTION / TRACKING =============================

@dataclass(frozen=True)
class MotionConfig:
    # Exponential smoothing of each tracked object's distance (1.0 = none).
    smoothing_alpha: float = 0.5
    # Movement is the least-squares trend of the raw distances over this
    # time window. Time-based, so it behaves the same at 3 fps or 30 fps.
    window_s: float = 1.2
    min_samples: int = 4
    min_span_s: float = 0.4
    # All three must hold before an object counts as approaching or receding:
    min_t_stat: float = 3.0         # trend must stand out from the jitter
    min_speed_mps: float = 0.3      # ignore slow drift
    max_ttc_s: float = 8.0          # distance / speed; ignore glacial change
    # ...and keep holding this long before the label switches to it.
    confirm_s: float = 0.4


@dataclass(frozen=True)
class TrackingConfig:
    iou_threshold: float = 0.25     # box overlap needed to be "the same object"
    max_age_s: float = 1.0          # forget an object unseen for this long


@dataclass(frozen=True)
class StabilityConfig:
    # A priority may rise at once, but only falls after staying lower for
    # this long, so a score wobbling around a cutoff does not flicker.
    downgrade_hold_s: float = 1.0


# ================================= ALERTS ===================================

@dataclass(frozen=True)
class AlertConfig:
    speak_priorities: tuple = ("HIGH", "MEDIUM")
    # Quiet time after an alert, keyed by the priority of that alert.
    cooldown_s: dict = field(default_factory=lambda: {"HIGH": 3.0, "MEDIUM": 5.0})
    # The same object at the same (or lower) priority is repeated at most
    # this often, so a parked car does not nag.
    same_object_repeat_s: float = 8.0
    # A different object that is HIGH may interrupt the cooldown.
    new_high_object_interrupts: bool = True
    # Append "2 more hazards nearby." when other HIGH/MEDIUM objects exist.
    mention_other_hazards: bool = True


# ================================= CAMERA ===================================

@dataclass(frozen=True)
class CameraConfig:
    # Pinhole focal length in pixels. Calibrate per device, see
    # core.distance.calibrate_focal_length.
    focal_length_px: float = 700.0
    # A box within this many pixels of the frame edge is treated as cut off.
    edge_margin_px: float = 2.0


# ================================== ROOT ====================================

@dataclass(frozen=True)
class HazardConfig:
    scoring: ScoringConfig = ScoringConfig()
    motion: MotionConfig = MotionConfig()
    tracking: TrackingConfig = TrackingConfig()
    stability: StabilityConfig = StabilityConfig()
    alerts: AlertConfig = AlertConfig()
    camera: CameraConfig = CameraConfig()

    def validate(self):
        """Raise ValueError when values contradict each other."""
        _validate(self)
        return self

    def to_dict(self):
        return asdict(self)


# =============================== VALIDATION =================================

def _validate(cfg):
    s = cfg.scoring
    problems = []

    if not s.medium_cutoff < s.high_cutoff:
        problems.append("scoring.medium_cutoff must be below scoring.high_cutoff")

    bands = s.distance_bands
    if not bands or bands[-1][0] is not None:
        problems.append("scoring.distance_bands must end with an open row [null, points, label]")
    else:
        bounds = [row[0] for row in bands[:-1]]
        if any(b is None or b <= 0 for b in bounds) or bounds != sorted(set(bounds)):
            problems.append("scoring.distance_bands upper bounds must be positive and increasing")
        points = [row[1] for row in bands]
        if points != sorted(points, reverse=True):
            # Keeps the promise "moving closer never lowers the score".
            problems.append("scoring.distance_bands points must not increase with distance")

    seen = {}
    for key, group in s.type_groups.items():
        for name in group.classes:
            if name in seen:
                problems.append(f"class '{name}' is in both '{seen[name]}' and '{key}'")
            seen[name] = key
        if group.distance_factor < 0:
            problems.append(f"type group '{key}' has a negative distance_factor")

    for key in s.position_points:
        if key not in POSITIONS:
            problems.append(f"scoring.position_points has unknown position '{key}'")
    mp = s.movement_points
    if set(mp) != set(MOVEMENTS):
        problems.append(f"scoring.movement_points needs exactly {list(MOVEMENTS)}")
    elif not mp["approaching"] >= mp["stationary"] >= mp["receding"]:
        problems.append("scoring.movement_points must satisfy approaching >= stationary >= receding")
    for alias, target in s.movement_synonyms.items():
        if target not in MOVEMENTS:
            problems.append(f"movement synonym '{alias}' points at unknown movement '{target}'")

    if not 0 <= s.low_confidence_below <= 1:
        problems.append("scoring.low_confidence_below must be between 0 and 1")
    if s.low_confidence_penalty > 0:
        problems.append("scoring.low_confidence_penalty must be zero or negative")
    if s.imminent_ttc_s is not None and s.imminent_ttc_s <= 0:
        problems.append("scoring.imminent_ttc_s must be positive or null")

    m = cfg.motion
    if not 0 < m.smoothing_alpha <= 1:
        problems.append("motion.smoothing_alpha must be in (0, 1]")
    if m.window_s <= 0 or m.min_span_s < 0 or m.min_span_s > m.window_s:
        problems.append("motion.window_s must be positive and at least motion.min_span_s")
    if m.min_samples < 3:
        problems.append("motion.min_samples must be at least 3")
    if m.min_speed_mps < 0 or m.max_ttc_s <= 0:
        problems.append("motion.min_speed_mps must be >= 0 and motion.max_ttc_s > 0")
    if m.min_t_stat < 0 or m.confirm_s < 0:
        problems.append("motion.min_t_stat and motion.confirm_s must not be negative")

    t = cfg.tracking
    if not 0 < t.iou_threshold <= 1:
        problems.append("tracking.iou_threshold must be in (0, 1]")
    if t.max_age_s <= 0:
        problems.append("tracking.max_age_s must be positive")

    if cfg.stability.downgrade_hold_s < 0:
        problems.append("stability.downgrade_hold_s must not be negative")

    a = cfg.alerts
    for p in a.speak_priorities:
        if p not in PRIORITY_LEVELS:
            problems.append(f"alerts.speak_priorities has unknown priority '{p}'")
    for p, seconds in a.cooldown_s.items():
        if p not in PRIORITY_LEVELS or seconds < 0:
            problems.append(f"alerts.cooldown_s['{p}'] is invalid")
    if a.same_object_repeat_s < 0:
        problems.append("alerts.same_object_repeat_s must not be negative")

    if cfg.camera.focal_length_px <= 0:
        problems.append("camera.focal_length_px must be positive")

    if problems:
        raise ValueError("Invalid hazard config:\n  - " + "\n  - ".join(problems))


# ================================ LOADING ===================================

def _merge_type_group(base, overrides, where):
    if not isinstance(overrides, dict):
        raise ValueError(f"{where} must be an object")
    known = {f.name for f in fields(TypeGroup)}
    for key in overrides:
        if key not in known:
            raise ValueError(f"Unknown setting {where}.{key}")
    values = dict(overrides)
    if "classes" in values:
        values["classes"] = tuple(str(c).strip().lower() for c in values["classes"])
    if base is None:
        missing = known - set(values)
        if missing:
            raise ValueError(f"New type group {where} needs {sorted(missing)}")
        return TypeGroup(**values)
    return replace(base, **values)


def _merge_value(current, new, where):
    if is_dataclass(current):
        return _merge_section(current, new, where)
    if where.endswith("type_groups"):
        merged = dict(current)
        for key, overrides in new.items():
            merged[key] = _merge_type_group(current.get(key), overrides, f"{where}.{key}")
        return merged
    if where.endswith("distance_bands"):
        return tuple(
            (None if row[0] is None else float(row[0]), float(row[1]), str(row[2]))
            for row in new
        )
    if isinstance(current, dict):
        if not isinstance(new, dict):
            raise ValueError(f"{where} must be an object")
        merged = dict(current)
        for key, value in new.items():
            key = key.upper() if where.endswith("cooldown_s") else key.lower()
            merged[key] = value
        return merged
    if isinstance(current, tuple):
        return tuple(new)
    return new


def _merge_section(section, overrides, where):
    if not isinstance(overrides, dict):
        raise ValueError(f"{where or 'config'} must be an object")
    known = {f.name: f for f in fields(section)}
    changes = {}
    for key, value in overrides.items():
        if key not in known:
            label = f"{where}.{key}" if where else key
            raise ValueError(f"Unknown setting {label}")
        path = f"{where}.{key}" if where else key
        changes[key] = _merge_value(getattr(section, key), value, path)
    return replace(section, **changes)


def config_from_dict(overrides, base=None):
    """Return ``base`` (default config) with ``overrides`` applied and checked."""
    base = base or DEFAULT_CONFIG
    return _merge_section(base, overrides or {}, "").validate()


def load_config(path=None, environ=None):
    """
    Load the hazard config.

    ``path`` wins; otherwise ``$VISIONSTRA_HAZARD_CONFIG``; otherwise defaults.
    """
    environ = os.environ if environ is None else environ
    path = path or environ.get(CONFIG_ENV_VAR)
    if not path:
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return config_from_dict(data)


DEFAULT_CONFIG = HazardConfig().validate()
