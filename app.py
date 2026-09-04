"""
AI-Based Camera Scene Detection & Auto-Enhancement App
--------------------------------------------------------
Backend: Flask
Computer Vision: OpenCV (Haar Cascade for face/portrait detection,
                  brightness/edge analysis for scene classification)
OCR: Tesseract (via pytesseract) for extracting text from document scenes
Database: SQLite (logs every processed image with detected scene + timestamp)

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000 on your laptop browser (use the same
network on your phone browser to open it there too, e.g. http://<your-pc-ip>:5000)
"""

import base64
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import pytesseract
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

DB_PATH = "scene_log.db"

# ---------- Database setup ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS scene_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_type TEXT,
            face_count INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_scene(scene_type, face_count=0):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO scene_log (scene_type, face_count, timestamp) VALUES (?, ?, ?)",
        (scene_type, face_count, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


# Haar cascade for face detection (ships with OpenCV, no download needed)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------- Scene detection logic ----------
def detect_scene(img):
    """Classify the image into a scene type using simple, explainable CV heuristics."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Brightness check -> low light
    brightness = np.mean(gray)

    # 2. Face detection -> portrait (supports multiple faces / group photos)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # 3. Edge density -> text/document (lots of sharp edges, low color variance)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # 4. Color variance -> landscape (colorful, few/no faces)
    saturation = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    avg_saturation = np.mean(saturation)

    if brightness < 70:
        return "low_light", faces
    elif len(faces) > 0:
        return "portrait", faces
    elif edge_density > 0.02 and avg_saturation < 50:
        return "document", faces
    else:
        return "landscape", faces


# ---------- Feature: Beauty filter (skin smoothing + warmth) for portraits ----------
# 4 selectable styles so the same photo can be re-processed differently on demand
BEAUTY_STYLES = {
    "natural":  {"d": 9,  "sigma": 60, "warm": 1.03, "cool": 0.98},
    "soft_glow": {"d": 15, "sigma": 90, "warm": 1.04, "cool": 0.97},
    "warm_tone": {"d": 9,  "sigma": 60, "warm": 1.12, "cool": 0.90},
    "high_contrast": {"d": 5, "sigma": 40, "warm": 1.02, "cool": 0.99},
}


def apply_beauty_filter(img, faces, style="natural"):
    """Applies skin smoothing + tone adjustment to detected face regions using the
    selected style, with feathered edges so the effect blends smoothly instead of
    showing a hard box."""
    params = BEAUTY_STYLES.get(style, BEAUTY_STYLES["natural"])
    result = img.copy().astype(np.float32)

    for (x, y, w, h) in faces:
        # Expand region slightly beyond the face box for a more natural blend
        pad_w, pad_h = int(w * 0.15), int(h * 0.15)
        x0, y0 = max(0, x - pad_w), max(0, y - pad_h)
        x1, y1 = min(img.shape[1], x + w + pad_w), min(img.shape[0], y + h + pad_h)

        face_roi = img[y0:y1, x0:x1]
        if face_roi.size == 0:
            continue

        # Bilateral filter smooths skin while preserving edges (eyes, lips, nose)
        smoothed = cv2.bilateralFilter(face_roi, d=params["d"], sigmaColor=params["sigma"], sigmaSpace=params["sigma"])

        if style == "high_contrast":
            # Boost local contrast instead of warmth, for a crisper look
            lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            smoothed = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        smoothed = smoothed.astype(np.float32)
        smoothed[:, :, 2] = np.clip(smoothed[:, :, 2] * params["warm"], 0, 255)  # R channel (BGR index 2)
        smoothed[:, :, 0] = np.clip(smoothed[:, :, 0] * params["cool"], 0, 255)  # B channel

        # Feathered (soft-edged) elliptical mask so the effect blends into the background
        # instead of showing a visible rectangle
        local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        cv2.ellipse(
            local_mask,
            ((x1 - x0) // 2, (y1 - y0) // 2),
            ((x1 - x0) // 2, (y1 - y0) // 2),
            0, 0, 360, 1.0, -1,
        )
        local_mask = cv2.GaussianBlur(local_mask, (31, 31), 0)
        local_mask_3ch = cv2.merge([local_mask, local_mask, local_mask])

        blended = face_roi.astype(np.float32) * (1 - local_mask_3ch) + smoothed * local_mask_3ch
        result[y0:y1, x0:x1] = blended

    return result.astype(np.uint8)


# ---------- Feature: Multi-face aware portrait enhancement ----------
def enhance_portrait(img, faces, beauty_style="natural"):
    """Blurs background while keeping ALL detected faces sharp (group photo support),
    then applies the selected beauty style to every face."""
    blurred = cv2.GaussianBlur(img, (35, 35), 0)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for (x, y, w, h) in faces:
        pad_w, pad_h = int(w * 0.6), int(h * 0.8)
        cv2.ellipse(
            mask,
            (x + w // 2, y + h // 2),
            (w // 2 + pad_w, h // 2 + pad_h),
            0, 0, 360, 255, -1,
        )
    mask_3ch = cv2.merge([mask, mask, mask])
    composited = np.where(mask_3ch == 255, img, blurred)
    beautified = apply_beauty_filter(composited, faces, style=beauty_style)
    return beautified


# ---------- Feature: OCR text extraction for document scenes ----------
def extract_text(img):
    """Runs OCR on a sharpened document image and returns cleaned extracted text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu thresholding gives Tesseract a clean black/white image to read
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        text = pytesseract.image_to_string(thresh)
    except Exception as e:
        text = ""
    return text.strip()


# ---------- Enhancement logic ----------
def enhance_image(img, scene_type, faces, beauty_style="natural"):
    if scene_type == "low_light":
        # Histogram equalization on the luminance channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    elif scene_type == "portrait":
        result = enhance_portrait(img, faces, beauty_style=beauty_style)

    elif scene_type == "document":
        # Sharpen for text clarity
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        result = cv2.filter2D(img, -1, kernel)

    else:  # landscape
        # Boost saturation/contrast slightly for vividness
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


def decode_image(data_url):
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image received"}), 400

    beauty_style = data.get("beauty_style", "natural")
    img = decode_image(data["image"])
    scene_type, faces = detect_scene(img)
    enhanced = enhance_image(img, scene_type, faces, beauty_style=beauty_style)

    extracted_text = ""
    if scene_type == "document":
        extracted_text = extract_text(enhanced)

    log_scene(scene_type, face_count=len(faces))

    return jsonify({
        "scene_type": scene_type,
        "face_count": int(len(faces)),
        "extracted_text": extracted_text,
        "original": encode_image(img),
        "enhanced": encode_image(enhanced),
    })


@app.route("/restyle", methods=["POST"])
def restyle():
    """Re-applies a different beauty style to an already-captured original image,
    without needing to take a new photo. Used by the style-switcher buttons."""
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image received"}), 400

    beauty_style = data.get("beauty_style", "natural")
    img = decode_image(data["image"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    enhanced = enhance_portrait(img, faces, beauty_style=beauty_style)

    return jsonify({
        "enhanced": encode_image(enhanced),
        "face_count": int(len(faces)),
    })


@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT scene_type, face_count, timestamp FROM scene_log ORDER BY id DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"scene_type": r[0], "face_count": r[1], "timestamp": r[2]} for r in rows])


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
