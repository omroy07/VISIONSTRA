"""
Read the few fields hazard prioritization needs from a detection dict.

Two shapes exist in this repository and both are accepted everywhere:

    /detect API  (backend/app.py)      {"object": "car", "distance_m": 5.0, ...}
    live camera  (detection /app.py)   {"label":  "Car", "distance":   5.0, ...}

Every reader returns None for missing or unusable values instead of raising.
"""

import math

POSITIONS = ("left", "center", "right")


def as_positive_float(value):
    """Return ``value`` as a finite float above zero, else None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def as_fraction(value):
    """Return ``value`` as a float in [0, 1], else None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0 <= number <= 1:
        return None
    return number


def clean_name(name):
    """Trimmed class name, or None when blank or not a string."""
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def normalise_name(name):
    """Lower-case class name used for table lookups ('' when blank)."""
    name = clean_name(name)
    return name.lower() if name else ""


def normalise_position(direction):
    """'left', 'center', 'right', or None."""
    if not isinstance(direction, str):
        return None
    key = direction.strip().lower()
    return key if key in POSITIONS else None


def detection_name(det):
    name = det.get("object")
    if name is None:
        name = det.get("label")
    return clean_name(name)


def raw_distance(det):
    """The detector's own distance estimate, before any smoothing."""
    value = det.get("distance_m")
    if value is None:
        value = det.get("distance")
    return as_positive_float(value)


def detection_distance(det):
    """The distance used for scoring: smoothed when available, else raw."""
    smoothed = as_positive_float(det.get("smoothed_distance_m"))
    return smoothed if smoothed is not None else raw_distance(det)


def detection_bbox(det):
    """(x1, y1, x2, y2) as floats with positive width and height, else None."""
    bbox = det.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)
