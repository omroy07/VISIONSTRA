import os
import re
import threading
from collections import OrderedDict

from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
from core.alerts import AlertPolicy
from core.direction import get_direction
from core.distance import estimate_distance_from_box
from core.hazard_config import load_config
from core.pipeline import HazardPrioritizer

app = Flask(__name__, static_folder="../frontend", static_url_path="")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = YOLO(os.path.join(_REPO_ROOT, "models", "yolov8n.pt"))

CONFIG = load_config()

# Each browser tab sends its own stream_id, so motion history and alert
# cooldowns never mix between viewers. Least recently used streams are
# dropped once there are too many.
_MAX_STREAMS = 16
_streams = OrderedDict()
_streams_lock = threading.Lock()
_STREAM_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _stream_state(stream_id):
    with _streams_lock:
        state = _streams.pop(stream_id, None)
        if state is None:
            state = (HazardPrioritizer(CONFIG), AlertPolicy(CONFIG.alerts))
        _streams[stream_id] = state
        while len(_streams) > _MAX_STREAMS:
            _streams.popitem(last=False)
        return state


# Home page
@app.route("/")
def index():
    return app.send_static_file("index.html")

# Detection API
@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("frame")
    if file is None:
        return jsonify({"error": "missing 'frame' image"}), 400
    img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "'frame' is not a decodable image"}), 400

    stream_id = request.form.get("stream_id", "default")
    if not _STREAM_ID.match(stream_id):
        return jsonify({"error": "invalid stream_id"}), 400

    h, w, _ = frame.shape
    results = model(frame, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            x_center = (x1 + x2) // 2

            detections.append({
                "object": label,
                "direction": get_direction(x_center, w),
                "distance_m": estimate_distance_from_box(
                    (x1, y1, x2, y2), label, w, h,
                    CONFIG.camera.focal_length_px, CONFIG.camera.edge_margin_px),
                "confidence": round(float(box.conf[0]), 2),
                "bbox": [x1, y1, x2, y2]
            })

    # Track, add movement, score, stabilise and rank.
    prioritizer, alert_policy = _stream_state(stream_id)
    ranked = prioritizer.process(detections)

    # At most one sentence per frame, attached to the hazard it describes.
    alert = alert_policy.consider(ranked)
    if alert is not None:
        ranked[0]["announcement"] = alert.text

    return jsonify(ranked)

if __name__ == "__main__":
    app.run(debug=True, port=5510)
