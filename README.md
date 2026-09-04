# 📸 AI Camera Scene Detection & Auto-Enhancement App

An intelligent camera application that automatically detects the scene type (portrait, low-light, document, or landscape) from a live camera feed and applies the appropriate real-time image enhancement — the same concept used in modern smartphone "AI Camera" modes.

## 🎯 Motivation

Smartphone camera quality is a key differentiator for mobile brands, especially in the budget/mid-range segment. This project explores how scene-aware auto-enhancement — a core feature in phone camera apps — can be built using lightweight, explainable computer vision techniques instead of heavy deep learning models, making it suitable for fast, on-device style processing.

## ✨ Features

- **Live camera capture** from any browser (laptop webcam or mobile browser) — no native app installation needed
- **Automatic scene detection** using OpenCV-based heuristics:
  - Low-light detection (brightness analysis)
  - Portrait detection (Haar Cascade face detection)
  - Document detection (edge density + saturation analysis)
  - Landscape (default/fallback case)
- **Scene-specific auto-enhancement**:
  - Low-light → CLAHE histogram equalization
  - Portrait → background blur (bokeh effect) while keeping face sharp
  - Document → sharpening filter for text clarity
  - Landscape → saturation/vibrance boost
- **Group photo support** — background blur and enhancement now work for *all* detected faces, not just one
- **Beauty filter** — automatic skin smoothing (bilateral filtering) and a subtle warm color tone applied to every detected face in portrait mode
- **OCR text extraction** — when a document scene is detected, the app runs Tesseract OCR on the sharpened image and displays the extracted text with a "Copy Text" button
- **Voice-controlled capture** — say "capture" or "take photo" and the browser's built-in speech recognition triggers the capture automatically (hands-free, no extra backend needed)
- **Before/after comparison** shown side-by-side
- **Detection history** logged to a local SQLite database with timestamps and face counts
- **Client-server architecture** — a thin browser-based client sends frames to a Flask backend that does all the processing

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Computer Vision | OpenCV (Haar Cascade, CLAHE, Canny edge detection, bilateral filtering) |
| OCR | Tesseract (via `pytesseract`) |
| Database | SQLite |
| Frontend | HTML5, JavaScript (`getUserMedia` API, Web Speech API) |

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ai-camera-scene-detection.git
cd ai-camera-scene-detection

# 2. Install dependencies
pip install -r requirements.txt

# 2a. OCR also needs the Tesseract engine installed on your system (not just the Python wrapper):
#     Windows: https://github.com/UB-Mannheim/tesseract/wiki (installer)
#     Mac:     brew install tesseract
#     Linux:   sudo apt install tesseract-ocr

# 3. Run the app
python app.py

# 4. Open in browser
# Visit http://127.0.0.1:5000
```

To access it from a mobile browser on the same WiFi network, find your machine's local IP (`ipconfig` on Windows / `ifconfig` on Mac/Linux) and visit `http://<your-ip>:5000` from your phone.

## 🧠 How Scene Detection Works

Instead of using a pretrained deep learning model (which requires internet access to download weights and more compute), this project uses simple, explainable image-processing rules:

1. **Low light**: average pixel brightness below a threshold
2. **Portrait**: a face is detected via Haar Cascade
3. **Document**: high edge density combined with low color saturation (text-like pattern)
4. **Landscape**: fallback case for colorful, non-face, non-text images

This approach is fast, fully offline, and mirrors a real constraint mobile manufacturers face: on-device camera AI needs to be lightweight and battery-efficient.

## 📈 Possible Extensions

- Replace rule-based classification with a lightweight pretrained CNN (e.g., MobileNet or MobileNet-SSD) for real-time object/product detection
- Convert the web frontend into a native Android app communicating with the same Flask API
- Deploy over HTTPS to avoid browser camera-permission restrictions on HTTP origins
- Add user accounts to track detection history per device

## 📂 Project Structure

```
├── app.py                  # Flask backend (scene detection + enhancement + DB logging)
├── templates/
│   └── index.html          # Camera capture frontend
├── requirements.txt        # Python dependencies
└── README.md
```

## 📝 License

This project is open-source and available for learning/demonstration purposes.
