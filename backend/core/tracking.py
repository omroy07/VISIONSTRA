"""
Give each detected object a stable ``track_id`` across frames.

A small greedy IoU tracker in pure Python. YOLO's own ``model.track()`` would
also work, but it installs extra packages at runtime and keeps its state
inside the model object; this one is dependency-free, deterministic and
unit-tested.

Matching rules, per frame:
  * a detection can only continue a track of the same class name;
  * the pair with the highest box overlap (IoU) is matched first, and pairs
    below ``iou_threshold`` are never matched;
  * each track and each detection is used at most once;
  * unmatched detections start new tracks; tracks unseen for longer than
    ``max_age_s`` are forgotten.

Detections without a usable ``bbox`` get ``track_id`` None and simply skip
the features that need history (movement, priority hold, repeat control).
"""

from .fields import detection_bbox, detection_name, normalise_name
from .hazard_config import DEFAULT_CONFIG


def iou(box_a, box_b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


class IouTracker:

    def __init__(self, config=None):
        self._cfg = config or DEFAULT_CONFIG.tracking
        self._tracks = {}          # track_id -> {"name", "bbox", "last_seen"}
        self._next_id = 1

    @property
    def active_ids(self):
        return set(self._tracks)

    def reset(self):
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections, now):
        """Return copies of ``detections`` with ``track_id`` set."""
        tagged = [dict(det) for det in detections or []]
        boxes = [detection_bbox(det) for det in tagged]
        names = [normalise_name(detection_name(det)) for det in tagged]

        pairs = []
        for d_index, box in enumerate(boxes):
            if box is None:
                continue
            for track_id, track in self._tracks.items():
                if track["name"] != names[d_index]:
                    continue
                overlap = iou(box, track["bbox"])
                if overlap >= self._cfg.iou_threshold:
                    pairs.append((-overlap, d_index, track_id))
        pairs.sort()

        used_tracks, assigned = set(), {}
        for _, d_index, track_id in pairs:
            if d_index in assigned or track_id in used_tracks:
                continue
            assigned[d_index] = track_id
            used_tracks.add(track_id)

        for d_index, det in enumerate(tagged):
            box = boxes[d_index]
            if box is None:
                det["track_id"] = None
                continue
            track_id = assigned.get(d_index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            self._tracks[track_id] = {
                "name": names[d_index], "bbox": box, "last_seen": now,
            }
            det["track_id"] = track_id

        expired = [tid for tid, track in self._tracks.items()
                   if now - track["last_seen"] > self._cfg.max_age_s]
        for tid in expired:
            del self._tracks[tid]

        return tagged
