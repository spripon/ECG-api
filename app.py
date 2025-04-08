
from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)
CORS(app)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def process_ecg(img):
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    original = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                          cv2.inRange(hsv, lower_red2, upper_red2))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 30, 150)
    combined = cv2.bitwise_or(mask, edges)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return original
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) == 4:
        pts = np.array([pt[0] for pt in approx], dtype="float32")
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        maxWidth = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        maxHeight = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(original, M, (maxWidth, maxHeight))
        return warped
    else:
        x, y, w, h = cv2.boundingRect(largest)
        return original[y:y + h, x:x + w]

@app.route("/", methods=["GET"])
def index():
    return "<h3>ECG API is running. Use POST /process with an image.</h3>"

@app.route("/process", methods=["POST"])
def process():
    if 'image' not in request.files:
        return {"error": "No image uploaded"}, 400
    file = request.files['image']
    data = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    processed = process_ecg(img)
    _, buffer = cv2.imencode(".jpg", processed)
    return send_file(BytesIO(buffer.tobytes()), mimetype='image/jpeg')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
