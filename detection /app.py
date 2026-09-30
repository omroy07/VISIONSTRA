import os
import sys
import cv2
import time
import numpy as np
import threading
from collections import deque
import pyttsx3
from flask import Flask, Response, render_template, jsonify, request
from ultralytics import YOLO
from keras_facenet import FaceNet
from numpy.linalg import norm
from threading import Lock

# =============================
# IMPORT SHARED RISK SCORER
# =============================
# Go up from "detection /" to the repository root, then into "backend".
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_BACKEND_DIR = os.path.join(_REPO_ROOT, "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.alerts import AlertPolicy
from core.direction import get_direction
from core.distance import estimate_distance_from_box
from core.hazard_config import load_config
from core.pipeline import HazardPrioritizer
from core.risk import score_single

# =============================
# APP SETUP
# =============================
app = Flask(__name__)

camera = None
camera_active = False
detected_objects = []
detection_lock = Lock()

# All weights, thresholds and timings: backend/core/hazard_config.py
# (override with a JSON file named by $VISIONSTRA_HAZARD_CONFIG).
CONFIG = load_config()

# Tracking, movement, scoring and ranking for the camera stream, and the
# rules for which hazard is spoken and when.
prioritizer = HazardPrioritizer(CONFIG)
alert_policy = AlertPolicy(CONFIG.alerts)

# Sentences actually spoken, newest first, for the dashboard.
spoken_alerts = deque(maxlen=20)
_speaking = threading.Event()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_DIR = os.path.join(BASE_DIR, "known_faces")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =============================
# LOAD MODELS
# =============================
yolo = YOLO(os.path.join(MODEL_DIR, "yolov8n.pt"))
embedder = FaceNet()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =============================
# TEXT TO SPEECH
# =============================
engine = pyttsx3.init()
engine.setProperty("rate", 165)

def speak(alert):
    """Voice one Alert from core.alerts on a background thread."""
    _speaking.set()
    spoken_alerts.appendleft({
        "time": time.strftime("%H:%M:%S"),
        "text": alert.text,
        "priority": alert.priority,
        "trigger": alert.trigger,
    })

    def _run():
        try:
            engine.say(alert.text)
            engine.runAndWait()
        finally:
            _speaking.clear()

    threading.Thread(target=_run, daemon=True).start()


# =============================
# UTILS
# =============================
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

def get_embedding(face):
    face = cv2.resize(face, (160, 160))
    face = np.expand_dims(face.astype("float32"), axis=0)
    return embedder.embeddings(face)[0]

# =============================
# LOAD KNOWN FACES
# =============================
known_embeddings = {}

if os.path.exists(KNOWN_DIR):
    for file in os.listdir(KNOWN_DIR):
        img = cv2.imread(os.path.join(KNOWN_DIR, file))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        known_embeddings[os.path.splitext(file)[0]] = get_embedding(img)

print("✅ Known Faces Loaded:", list(known_embeddings.keys()))

def recognize_face(face):
    emb = get_embedding(face)
    identity = "Unknown"
    min_dist = 1.0

    for name, known_emb in known_embeddings.items():
        dist = cosine_distance(emb, known_emb)
        if dist < min_dist:
            min_dist = dist
            identity = name

    return identity if min_dist < 0.6 else "Unknown"

# =============================
# PRIORITY BOX COLORS (BGR for OpenCV)
# =============================
_BOX_COLORS = {
    "HIGH":   (68, 68, 239),    # #ef4444 → BGR
    "MEDIUM": (11, 158, 245),   # #f59e0b → BGR
    "LOW":    (128, 128, 107),  # #6b7280 → BGR
}

# =============================
# VIDEO STREAM
# =============================
def generate_frames():
    global camera, camera_active

    while True:
        if not camera_active or camera is None:
            time.sleep(0.1)
            continue

        success, frame = camera.read()
        if not success:
            break

        h, w, _ = frame.shape

        # TEMP list for this frame
        current_detections = []

        # YOLO DETECTION
        results = yolo(frame, stream=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = yolo.names[int(box.cls[0])]

                direction = get_direction((x1 + x2) // 2, w)
                distance = estimate_distance_from_box(
                    (x1, y1, x2, y2), label, w, h,
                    CONFIG.camera.focal_length_px, CONFIG.camera.edge_margin_px)

                # ✅ Store detection
                current_detections.append({
                    "label": label.capitalize(),
                    "direction": direction,
                    "distance": distance,
                    "confidence": round(float(box.conf[0]), 2),
                    "bbox": [x1, y1, x2, y2]
                })

        # Track, add movement, score, stabilise and rank.
        ranked = prioritizer.process(current_detections)

        # ✅ Draw boxes with priority-colored labels on the video frame
        for det in ranked:
            bbox = det.get("bbox")
            if bbox:
                bx1, by1, bx2, by2 = bbox
                priority = det.get("priority", "LOW")
                color = _BOX_COLORS.get(priority, _BOX_COLORS["LOW"])
                rank_num = det.get("rank", "")
                pri_tag = f"[{priority}]"
                rank_tag = " Primary" if rank_num == 1 else ""
                distance_value = det.get("smoothed_distance_m") or det.get("distance")
                distance_text = f"{distance_value:.1f}m" if distance_value else "distance unknown"

                # Draw bounding box
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(
                    frame,
                    f"{det['label']} {det['direction']} {distance_text} {pri_tag}{rank_tag}",
                    (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

        # ✅ Update shared detections
        with detection_lock:
            detected_objects[:] = ranked

        # 🔊 At most one sentence: the primary hazard, when the policy allows
        alert = alert_policy.consider(ranked, speaking=_speaking.is_set())
        if alert is not None:
            speak(alert)

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


# =============================
# ROUTES
# =============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detections")
def detections():
    with detection_lock:
        return jsonify(list(detected_objects))

@app.route("/alerts")
def alerts():
    return jsonify(list(spoken_alerts))

@app.route("/start_camera")
def start_camera():
    global camera, camera_active

    if not camera_active:
        camera = cv2.VideoCapture(0)
        camera_active = True

    return jsonify({"status": "started"})

@app.route("/stop_camera")
def stop_camera():
    global camera, camera_active

    camera_active = False
    time.sleep(0.2)

    if camera:
        camera.release()
        camera = None

    with detection_lock:
        detected_objects.clear()
    prioritizer.reset()
    alert_policy.reset()

    return jsonify({"status": "stopped"})

# =============================
# TEST DEMO ROUTES
# =============================
@app.route("/test_demo")
def test_demo():
    return render_template("test_demo.html")

@app.route("/api/score", methods=["POST"])
def api_score():
    """
    JSON {name, distance, direction, movement, confidence, time_to_contact_s}
    → score, priority, reason and per-factor breakdown. Every field is
    optional; the scorer treats blank or invalid values as unknown.
    """
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "expected a JSON object"}), 400

    result = score_single(
        data.get("name") or None,
        data.get("distance"),
        data.get("direction") or None,
        data.get("movement") or None,
        confidence=data.get("confidence"),
        time_to_contact_s=data.get("time_to_contact_s"),
        config=CONFIG.scoring,
    )
    result["cutoffs"] = {"HIGH": CONFIG.scoring.high_cutoff,
                         "MEDIUM": CONFIG.scoring.medium_cutoff}
    return jsonify(result)

# =============================
# RUN
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5515, debug=False, threaded=True)
