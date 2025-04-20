"""
ECG-API – GPT-4o Vision + fallback couleur
© spripon – mai 2025
"""

import os, base64, json, cv2, numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VISION_MODEL = os.getenv("MODEL_NAME", "gpt-4o")
MAX_DIM = 1600
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
CORS(app)

# ────── utilitaires ──────
def order_pts(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s, d = pts.sum(axis=1), np.diff(pts, axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

def warp(roi, poly):
    pts = np.array([[p['x'], p['y']] for p in poly], dtype="float32")
    if len(pts) != 4:
        pts = cv2.boxPoints(cv2.minAreaRect(pts)).astype("float32")
    rect = order_pts(pts)
    (tl, tr, br, bl) = rect
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(roi, M, (W, H))

# ────── Vision detection ──────
def detect_vision(img):
    if client is None:
        return None
    h, w = img.shape[:2]; scale = 1
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img_r = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        img_r = img
    ok, buf = cv2.imencode(".png", img_r)
    b64 = base64.b64encode(buf).decode()

    messages = [{"role": "user", "content": [
        {"type": "text", "text": (
            "Localise exclusivement la zone quadrillée rose/rougeâtre ECG. "
            "Ignore toute marge blanche et autre élément externe. Réponds JSON {\"points\":[{\"x\":int,\"y\":int}]} (4 à 8 coins)."
        )},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    ]}]

    tools = [{"type": "function", "function": {"name": "set", "parameters": {
        "type": "object", "properties": {
            "points": {"type": "array", "items": {
                "type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"]
            }, "minItems": 4, "maxItems": 8}
        }, "required": ["points"]}}}]

    rsp = client.chat.completions.create(
        model=VISION_MODEL, messages=messages, tools=tools,
        tool_choice={"type": "function", "function": {"name": "set"}}, temperature=0)

    pts = rsp.choices[0].message.tool_calls[0].function.arguments
    if isinstance(pts, str): pts = json.loads(pts)
    pts = pts["points"]
    xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
    return {"poly": [{"x": int(x/scale), "y": int(y/scale)} for x, y in zip(xs, ys)]}

# ────── Fallback couleur ──────
def detect_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0, 30, 50]), np.array([20, 255, 255])
    lower2, upper2 = np.array([160, 30, 50]), np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c); box = cv2.boxPoints(rect).astype(int)
    return {"poly": [{"x": int(x), "y": int(y)} for x, y in box]}

# ────── amélioration NB ──────
def enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (16, 16)).apply(l)
    norm = cv2.divide(cv2.merge((l, a, b)), cv2.blur(cv2.merge((l, a, b)), (101, 101)), scale=255)
    gray = cv2.cvtColor(norm, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 7)
    if np.mean(bw[:30, :30]) < 128: bw = cv2.bitwise_not(bw)
    return bw

# ────── pipeline ──────
def process(img):
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    box = detect_vision(img)
    if not box:
        box = detect_color(img)

    roi = warp(img, box["poly"])
    return enhance(roi)

# ────── Flask route ──────
@app.route("/process", methods=["POST"])
def handle():
    if "image" not in request.files:
        return jsonify(error="image manquante"), 400
    data = request.files["image"].read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    out = process(img)
    _, buf = cv2.imencode(".png", out)
    return send_file(BytesIO(buf.tobytes()), mimetype="image/png")

@app.route("/")
def home(): return "ECG-API Vision + couleur"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)
