"""
Hazard risk scoring and ranking.

Pure Python: no Flask, OpenCV, YOLO or NumPy. Callers pass plain values or
detection dicts and get enriched copies back. Every number comes from
``core.hazard_config``; nothing is tuned in this file.

For one object:

    score = type_weight
          + distance_points × distance_factor
          + position_points
          + movement_points
          + confidence_penalty
    clamped to 0–100, rounded to one decimal.

    HIGH   score >= high_cutoff   (or an approaching object that is about to
                                   arrive, see ``imminent_ttc_s``)
    MEDIUM score >= medium_cutoff
    LOW    otherwise

Each result carries the per-factor ``breakdown`` and a plain-English
``reason`` so the priority can always be explained.
"""

from .fields import (
    as_fraction,
    as_positive_float,
    clean_name,
    detection_distance,
    detection_name,
    normalise_name,
    normalise_position,
)
from .hazard_config import DEFAULT_CONFIG, PRIORITY_LEVELS

_SCORE_MIN = 0.0
_SCORE_MAX = 100.0


# ================================ HELPERS ===================================

def priority_level(priority):
    """0 for LOW, 1 for MEDIUM, 2 for HIGH (unknown values count as LOW)."""
    try:
        return PRIORITY_LEVELS.index(priority)
    except ValueError:
        return 0


def classify(score, config=None):
    """Map a rounded score to LOW / MEDIUM / HIGH."""
    cfg = config or DEFAULT_CONFIG.scoring
    if score >= cfg.high_cutoff:
        return "HIGH"
    if score >= cfg.medium_cutoff:
        return "MEDIUM"
    return "LOW"


def type_group_for(name, config=None):
    """Return (group_key, TypeGroup) for a class name."""
    cfg = config or DEFAULT_CONFIG.scoring
    key = normalise_name(name)
    if not key:
        return "unknown", cfg.unknown_group
    for group_key, group in cfg.type_groups.items():
        if key in group.classes:
            return group_key, group
    return "other", cfg.other_group


def _distance_band(distance, cfg):
    """(points before factor, words) for a checked distance or None."""
    if distance is None:
        return cfg.distance_missing_points, None
    for upper, points, words in cfg.distance_bands:
        if upper is None or distance <= upper:
            return points, words
    # Unreachable with a validated config: the last band is open-ended.
    return cfg.distance_bands[-1][1], cfg.distance_bands[-1][2]


def _movement_key(movement, cfg):
    if not isinstance(movement, str):
        return None
    key = movement.strip().lower()
    key = cfg.movement_synonyms.get(key, key)
    return key if key in cfg.movement_points else None


def _format_metres(distance):
    text = f"{distance:.1f}".rstrip("0").rstrip(".")
    return text if text not in ("", "0") else "0.1"


# ============================= REASON TEXT ==================================

_MOVEMENT_WORDS = {
    "approaching": "it is getting closer",
    "receding": "it is moving away",
    "stationary": "its distance is holding steady",
}


def _build_reason(display, group, distance, band_words, position, movement,
                  confidence, low_confidence, imminent_ttc):
    facts = [display[0].upper() + display[1:] if display else "Unknown object"]
    facts.append(f"{_format_metres(distance)} m" if distance is not None
                 else "distance unknown")
    facts.append(position or "position unknown")
    if movement:
        facts.append(movement)

    why = [f"it is {group.description}"]
    why.append(f"it is {band_words}" if distance is not None
               else "distance is unknown")
    if position == "center":
        why.append("it is in the walking line")
    elif position is None:
        why.append("position is unknown")
    if movement:
        why.append(_MOVEMENT_WORDS[movement])
    if low_confidence:
        why.append(f"the detector is unsure ({confidence:.0%} confidence)")

    sentence = "; ".join(why)
    text = ", ".join(facts) + ". " + sentence[0].upper() + sentence[1:] + "."
    if imminent_ttc is not None:
        text += (f" Raised to HIGH: about {imminent_ttc:.1f} s"
                 " until it reaches the camera.")
    return text


# ============================== PUBLIC API ==================================

def score_single(name=None, distance=None, direction=None, movement=None,
                 *, confidence=None, time_to_contact_s=None, config=None):
    """
    Score one detected object.

    Parameters
    ----------
    name : str or None           YOLO class name, any case ("person", "Bicycle").
    distance : number or None    Estimated metres. Missing, non-numeric,
                                 non-finite or <= 0 counts as unknown.
    direction : str or None      "Left", "Center" or "Right".
    movement : str or None       "approaching", "stationary", "receding"
                                 (or a synonym from the config).
    confidence : float or None   Detector confidence in [0, 1].
    time_to_contact_s : float    Seconds until an approaching object arrives.
    config : ScoringConfig       Defaults to ``DEFAULT_CONFIG.scoring``.

    Returns
    -------
    dict with risk_score, priority, reason, category, breakdown, imminent.
    """
    cfg = config or DEFAULT_CONFIG.scoring

    display = clean_name(name)
    category, group = type_group_for(display, cfg)
    distance = as_positive_float(distance)
    band_points, band_words = _distance_band(distance, cfg)
    position = normalise_position(direction)
    movement = _movement_key(movement, cfg)
    confidence = as_fraction(confidence)
    low_confidence = confidence is not None and confidence < cfg.low_confidence_below

    breakdown = {
        "type": float(group.weight),
        "distance": round(band_points * group.distance_factor, 1),
        "position": float(cfg.position_points.get(position, cfg.position_missing_points)
                          if position else cfg.position_missing_points),
        "movement": float(cfg.movement_points[movement] if movement
                          else cfg.movement_missing_points),
        "confidence": float(cfg.low_confidence_penalty if low_confidence else 0),
    }
    raw = (group.weight + band_points * group.distance_factor
           + breakdown["position"] + breakdown["movement"] + breakdown["confidence"])
    score = round(min(max(raw, _SCORE_MIN), _SCORE_MAX), 1)
    priority = classify(score, cfg)

    ttc = as_positive_float(time_to_contact_s)
    imminent = (
        cfg.imminent_ttc_s is not None
        and movement == "approaching"
        and ttc is not None
        and ttc <= cfg.imminent_ttc_s
        and not low_confidence
    )
    if imminent:
        priority = "HIGH"

    reason = _build_reason(display, group, distance, band_words, position,
                           movement, confidence, low_confidence,
                           ttc if imminent else None)

    return {
        "risk_score": score,
        "priority": priority,
        "reason": reason,
        "category": category,
        "breakdown": breakdown,
        "imminent": imminent,
    }


def score_detection(det, config=None):
    """Return a copy of one detection dict with the score fields added."""
    result = score_single(
        detection_name(det),
        detection_distance(det),
        det.get("direction"),
        det.get("movement"),
        confidence=det.get("confidence"),
        time_to_contact_s=det.get("time_to_contact_s"),
        config=config,
    )
    enriched = dict(det)
    enriched.update(result)
    return enriched


def score_detections(detections, config=None):
    """Score every detection. Order is unchanged and inputs are not mutated."""
    return [score_detection(det, config) for det in detections or []]


def _sort_key(item):
    index, det = item
    distance = detection_distance(det)
    position = normalise_position(det.get("direction"))
    return (
        -priority_level(det.get("priority")),       # 1. higher priority
        -det.get("risk_score", 0.0),                # 2. higher score
        (0, distance) if distance is not None else (1, 0.0),  # 3. nearer; unknown last
        0 if position == "center" else 1 if position else 2,  # 4. center, side, unknown
        normalise_name(detection_name(det)),        # 5. name A–Z
        index,                                      # 6. detector order
    )


def sort_and_rank(scored):
    """
    Sort already-scored detections from most to least urgent and set
    ``rank`` (1 = the primary hazard). Returns a new list.
    """
    ordered = [det for _, det in sorted(enumerate(scored), key=_sort_key)]
    for rank, det in enumerate(ordered, start=1):
        det["rank"] = rank
    return ordered


def rank_detections(detections, config=None):
    """Score, sort and rank a list of detection dicts in one stateless call."""
    return sort_and_rank(score_detections(detections, config))
