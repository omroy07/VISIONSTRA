"""
Monocular distance estimation from a bounding box (pinhole camera model):

    distance_m = real_size_m × focal_length_px ÷ size_in_pixels

The old version assumed every object was 0.5 m wide. A car is ~1.8 m wide,
so cars were estimated ~3.5× closer than they really were and won almost
every ranking. Each class now has its own reference size.

Height is preferred because it does not change when an object turns (a car
seen side-on is ~4.5 m long but still ~1.5 m tall). If the box touches the
top or bottom of the frame the object is cut off, so width is used instead.
If both are cut off, the nearer of the two estimates is used (the safe side).
"""

from .hazard_config import DEFAULT_CONFIG

FOCAL_LENGTH = DEFAULT_CONFIG.camera.focal_length_px   # kept for old imports
KNOWN_WIDTH = 0.5                                      # metres, fallback width

# class name -> (typical width m, typical height m or None)
REFERENCE_SIZES_M = {
    "person":        (0.50, 1.70),
    "bicycle":       (0.60, 1.05),
    "motorcycle":    (0.80, 1.15),
    "car":           (1.80, 1.50),
    "bus":           (2.55, 3.20),
    "truck":         (2.50, 3.00),
    "train":         (3.00, 3.80),
    "dog":           (0.30, 0.60),
    "cat":           (0.20, 0.30),
    "horse":         (0.60, 1.60),
    "cow":           (0.70, 1.40),
    "sheep":         (0.50, 0.80),
    "bench":         (1.50, 0.85),
    "chair":         (0.50, 0.90),
    "fire hydrant":  (0.35, 0.75),
    "stop sign":     (0.75, 0.75),
    "potted plant":  (0.50, 0.70),
}


def estimate_distance(bbox_width, real_width_m=KNOWN_WIDTH, focal_length_px=FOCAL_LENGTH):
    """Distance from box width alone. Returns None for a non-positive width."""
    if bbox_width is None or bbox_width <= 0:
        return None
    return (real_width_m * focal_length_px) / bbox_width


def estimate_distance_from_box(bbox, label, frame_width, frame_height,
                               focal_length_px=None, edge_margin_px=None):
    """
    Distance in metres (rounded to 0.01) for one YOLO box, or None when the
    box is unusable.
    """
    camera = DEFAULT_CONFIG.camera
    focal = focal_length_px or camera.focal_length_px
    margin = camera.edge_margin_px if edge_margin_px is None else edge_margin_px

    x1, y1, x2, y2 = bbox
    width_px, height_px = x2 - x1, y2 - y1
    if width_px <= 0 or height_px <= 0:
        return None

    key = label.strip().lower() if isinstance(label, str) else ""
    real_width, real_height = REFERENCE_SIZES_M.get(key, (KNOWN_WIDTH, None))

    cut_vertically = y1 <= margin or y2 >= frame_height - margin
    cut_horizontally = x1 <= margin or x2 >= frame_width - margin

    from_width = real_width * focal / width_px
    from_height = real_height * focal / height_px if real_height else None

    if from_height is None:
        distance = from_width
    elif not cut_vertically:
        distance = from_height
    elif not cut_horizontally:
        distance = from_width
    else:
        distance = min(from_width, from_height)
    return round(distance, 2)


def calibrate_focal_length(pixel_size, distance_m, real_size_m):
    """
    Focal length in pixels from one measurement: stand an object of known
    size at a known distance, read its box size in pixels, and put the result
    in ``camera.focal_length_px`` of the hazard config.
    """
    if pixel_size <= 0 or distance_m <= 0 or real_size_m <= 0:
        raise ValueError("pixel_size, distance_m and real_size_m must be positive")
    return pixel_size * distance_m / real_size_m
