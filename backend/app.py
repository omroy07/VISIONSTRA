from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
from core.direction import get_direction
from core.distance import estimate_distance

app = Flask(__name__, static_folder="../frontend", static_url_path="")

model = YOLO("models/yolov8n.pt")

# Home page
@app.route("/")
def index():
    return app.send_static_file("index.html")

# Detection API
@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["frame"]
    img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    h, w, _ = frame.shape
    results = model(frame)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            x_center = (x1 + x2) // 2
            bbox_width = x2 - x1

            detections.append({
                "object": model.names[cls],
                "direction": get_direction(x_center, w),
                "distance_m": estimate_distance(bbox_width),
                "bbox": [x1, y1, x2, y2]
            })

    return jsonify(detections)

if __name__ == "__main__":
    app.run(debug=True, port=5510)
