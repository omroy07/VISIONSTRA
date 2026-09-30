import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
KNOWN_DIR = os.path.join(BASE_DIR, "known_faces")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ORB feature extractor
orb = cv2.ORB_create(nfeatures=1000)

# Load known images
known_descriptors = {}

for file in os.listdir(KNOWN_DIR):
    path = os.path.join(KNOWN_DIR, file)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        continue

    name = os.path.splitext(file)[0]
    known_descriptors[name] = des

print("Loaded identities:", known_descriptors.keys())

# Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def recognize_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    if des is None:
        return "Unknown", 0

    best_score = 0
    identity = "Unknown"

    for name, known_des in known_descriptors.items():
        matches = bf.match(des, known_des)
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = [m for m in matches if m.distance < 60]
        score = len(good_matches)

        if score > best_score:
            best_score = score
            identity = name

    if best_score < 20:
        identity = "Unknown"

    return identity, best_score

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"recognized": False})

    file = request.files["image"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        return jsonify({"recognized": False})

    name, score = recognize_image(img)

    return jsonify({
        "recognized": True,
        "identity": name,
        "match_score": score
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5520, debug=True)
