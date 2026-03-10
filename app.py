"""
Student Face Detection & Attendance - OpenCV LBPH (100% offline, no model downloads)
"""
import csv
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
ATTENDANCE_FILE = BASE_DIR / "attendance.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "lbph_model.yml"
NAMES_FILE = MODELS_DIR / "face_names.pkl"
TEMP_DIR = BASE_DIR / "temp"
FACE_SIZE = (200, 200)  # Resize face to this for LBPH

MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Haar cascade for face detection (bundled with OpenCV)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

_last_marked = {}
COOLDOWN_SEC = 10


def get_faces_gray(image_path):
    """Detect faces and return list of (gray_face, bbox)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
    result = []
    for (x, y, w, h) in faces:
        face_roi = gray[y : y + h, x : x + w]
        face_resized = cv2.resize(face_roi, FACE_SIZE)
        result.append(face_resized)
    return result


def load_model():
    """Load LBPH model and name list."""
    names = []
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if MODEL_FILE.exists():
        recognizer.read(str(MODEL_FILE))
    if NAMES_FILE.exists() and NAMES_FILE.stat().st_size > 0:
        with open(NAMES_FILE, "rb") as f:
            names = pickle.load(f)
    return recognizer, names


def save_model(recognizer, names):
    recognizer.write(str(MODEL_FILE))
    with open(NAMES_FILE, "wb") as f:
        pickle.dump(names, f)


def mark_attendance(name):
    now = datetime.now()
    last = _last_marked.get(name)
    if last and (now - last).total_seconds() < COOLDOWN_SEC:
        return False
    _last_marked[name] = now
    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    return True


def train_face(name, image_path):
    """Train LBPH with new face."""
    faces = get_faces_gray(image_path)
    if not faces:
        raise ValueError("No face detected. Use good lighting and look straight at the camera.")
    recognizer, names = load_model()
    faces_data_path = MODELS_DIR / "faces_data.pkl"
    all_faces, all_labels = [], []
    if faces_data_path.exists():
        with open(faces_data_path, "rb") as f:
            all_faces, all_labels = pickle.load(f)

    if name in names:
        idx = names.index(name)
        all_faces = [f for i, f in enumerate(all_faces) if all_labels[i] != idx]
        all_labels = [l for l in all_labels if l != idx]
    else:
        idx = len(names)
        names.append(name)

    for face in faces:
        all_faces.append(face)
        all_labels.append(idx)

    labels = np.array(all_labels, dtype=np.int32)
    recognizer.train(all_faces, labels)
    save_model(recognizer, names)
    with open(faces_data_path, "wb") as f:
        pickle.dump((all_faces, all_labels), f)


def recognize_faces(image_path):
    """Recognize faces; return list of names."""
    recognizer, names = load_model()
    if not names:
        return []
    faces = get_faces_gray(image_path)
    if not faces:
        return []
    result = []
    for face in faces:
        label, confidence = recognizer.predict(face)
        # LBPH: lower confidence = better match; typically < 70 is good
        if 0 <= label < len(names) and confidence < 80:
            result.append(names[label])
    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "GET":
        return render_template("train.html")

    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"message": "Name is required"}), 400

    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"message": "Image is required"}), 400

    ext = Path(file.filename).suffix if file.filename else ".jpg"
    if ext.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        ext = ".jpg"
    temp_path = (TEMP_DIR / f"train{ext}").resolve()
    try:
        file.save(str(temp_path))
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            return jsonify({"message": "Failed to save image."}), 400
        train_face(name, temp_path)
        return jsonify({"message": f"Face trained successfully for '{name}'"})
    except ValueError as e:
        return jsonify({"message": str(e)}), 400
    except Exception as e:
        app.logger.exception("Train failed")
        return jsonify({"message": str(e)}), 500
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if request.method == "GET":
        return render_template("attendance.html")

    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"names": [], "message": "Image is required"}), 400

    temp_path = TEMP_DIR / "attendance.jpg"
    try:
        file.save(temp_path)
        names = recognize_faces(temp_path)
        marked = [n for n in names if mark_attendance(n)]
        return jsonify({"names": names, "marked": marked})
    except Exception as e:
        return jsonify({"names": [], "message": str(e)}), 500
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.route("/view")
def view_attendance():
    rows = []
    if ATTENDANCE_FILE.exists():
        with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    rows.append([row[0], row[1], row[2]])
                elif len(row) == 2:
                    try:
                        dt = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
                        rows.append([row[0], dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")])
                    except ValueError:
                        rows.append([row[0], row[1], ""])
    return render_template("view_attendance.html", rows=rows)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
