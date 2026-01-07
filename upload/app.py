import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 🗂 Target folder to save images
BASE_DIR = r"D:\MLPROJECTS\visionstra\detection\known_faces"

# ✅ Allowed image types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_face():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        relation = request.form.get("relation", "").strip()
        file = request.files.get("image")

        if not name or not relation or not file:
            return "❌ All fields are required"

        if not allowed_file(file.filename):
            return "❌ Invalid image format (only png/jpg/jpeg)"

        # Create person folder
        person_folder = os.path.join(BASE_DIR, name.replace(" ", "_"))
        os.makedirs(person_folder, exist_ok=True)

        # Save image
        filename = secure_filename(file.filename)
        file.save(os.path.join(person_folder, filename))

        # Save info.txt
        with open(os.path.join(person_folder, "info.txt"), "w") as f:
            f.write(f"Name: {name}\nRelation: {relation}")

        return "✅ Image uploaded and saved successfully!"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5505)
