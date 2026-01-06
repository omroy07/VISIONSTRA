import os
import cv2
import time
import numpy as np
import threading
import pyttsx3
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
from keras_facenet import FaceNet
from numpy.linalg import norm
from threading import Lock

# =============================
# APP SETUP
# =============================
app = Flask(__name__)

camera = None
camera_active = False
detected_objects = []
detection_lock = Lock()

last_spoken_time = {}
VOICE_COOLDOWN = 4  # seconds



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
last_spoken = ""

def speak(text):
    now = time.time()

    if text in last_spoken_time:
        if now - last_spoken_time[text] < VOICE_COOLDOWN:
            return

    last_spoken_time[text] = now
    engine.say(text)
    engine.runAndWait()


# =============================
# UTILS
# =============================
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

def get_direction(x_center, frame_width):
    if x_center < frame_width / 3:
        return "Left"
    elif x_center > 2 * frame_width / 3:
        return "Right"
    return "Center"

def estimate_distance(bbox_width):
    if bbox_width <= 0:
        return 0
    return round((0.5 * 600) / bbox_width, 2)

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
# VIDEO STREAM
# =============================
def generate_frames():
    global camera, camera_active, detected_objects

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
                distance = estimate_distance(x2 - x1)

                # ✅ Store detection
                current_detections.append({
                    "label": label.capitalize(),
                    "direction": direction,
                    "distance": distance
                })

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {direction} {distance}m",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # ✅ Update shared detections ONLY if something detected
        if current_detections:
            detected_objects[:] = current_detections

            # 🔊 Speak summary (only first detection to avoid noise)
            first = current_detections[0]

            voice_text = (
                f"{first['label']} detected on "
                f"{first['direction'].lower()} at "
                f"{first['distance']} meters"
            )

            threading.Thread(
                target=speak,
                args=(voice_text,),
                daemon=True
            ).start()

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
    return jsonify(detected_objects)

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

    return jsonify({"status": "stopped"})

# =============================
# RUN
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5515, debug=False, threaded=True)
