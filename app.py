"""
API ECG – Flask + OpenCV + OpenAI Vision (o3 / o4‑mini)
© spripon 2025
"""
import os, base64, cv2, numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI

# ─────────────────────  CONFIG  ──────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # ajoutez‑le sur Railway
VISION_MODEL   = os.getenv("MODEL_NAME", "gpt-4o-mini-vision-preview")
MAX_DIM_VISION = int(os.getenv("VISION_MAX_DIM", "1600"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ────────────────────  FLASK APP  ────────────────────
app = Flask(__name__)
CORS(app)      # CORS large pour Lovable

# ───────────────────  OUTILS GÉNÉRIQUES  ─────────────
def order_points(pts: np.ndarray) -> np.ndarray:
    """pivote les 4 coins pour qu'ils soient tl,tr,br,bl"""
    rect = np.zeros((4, 2), dtype="float32")
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

# ───────────────────  OPENAI VISION  ─────────────────
def detect_grid_openai(img_bgr: np.ndarray) -> dict | None:
    """
    Utilise GPT‑4o(o4‑mini) pour renvoyer {x,y,w,h} (bbox du quadrillage)
    Retourne None si échec ou si la clé API n'est pas définie.
    """
    if client is None:
        return None

    h0, w0 = img_bgr.shape[:2]
    scale  = 1.0
    if max(h0, w0) > MAX_DIM_VISION:
        scale = MAX_DIM_VISION / max(h0, w0)
        img_r = cv2.resize(img_bgr, (int(w0*scale), int(h0*scale)))
    else:
        img_r = img_bgr

    ok, buf = cv2.imencode(".png", img_r)
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode()

    messages = [{
        "role": "user",
        "content": [
            { "type": "text",
              "text": (
                "Localise la plus grande zone contenant le quadrillage ECG "
                "(rose, jaune ou orange) avec les dérivations D1‑D3, aVR‑aVF, V1‑V6. "
                "Réponds uniquement par JSON : {\"x\":int,\"y\":int,\"w\":int,\"h\":int} "
                "sur l'image que je t'envoie."
              )},
            { "type": "image_url",
              "image_url": { "url": f"data:image/png;base64,{b64}" } }
        ]
    }]
    tools = [{
        "type":"function",
        "function":{
            "name":"set_bbox",
            "parameters":{
                "type":"object",
                "properties":{
                    "x":{"type":"integer"},
                    "y":{"type":"integer"},
                    "w":{"type":"integer"},
                    "h":{"type":"integer"}
                },
                "required":["x","y","w","h"]
            }
        }
    }]

    try:
        rsp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="set_bbox",
            temperature=0
        )
        bbox = rsp.choices[0].message.tool_calls[0].arguments
        # remap sur image originale
        for k in bbox:
            bbox[k] = int(bbox[k] / scale)
        return bbox
    except Exception as e:
        print("Vision API error:", e)
        return None

# ───────────────────  PIPELINE ECG  ──────────────────
def hsv_fallback(img_bgr: np.ndarray) -> np.ndarray:
    """Méthode actuelle basée couleur (prise de votre script d'origine)."""
    img_blur = cv2.GaussianBlur(img_bgr, (5,5), 0)
    hsv      = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    min_sat, min_val = 20, 90
    ranges = [
        ([0,   min_sat, min_val], [15, 255, 255]),   # rose/rouge
        ([165, min_sat, min_val], [180,255, 255]),   # rose/rouge
        ([16,  min_sat, min_val], [45, 255, 255])    # orange/jaune
    ]
    mask = None
    for low, high in ranges:
        cur = cv2.inRange(hsv, np.array(low), np.array(high))
        mask = cur if mask is None else cv2.bitwise_or(mask, cur)

    kernel = np.ones((7,7), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr  # fallback total
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return img_bgr[y:y+h, x:x+w]

def perspective_or_bbox(roi: np.ndarray):
    """essaie de redresser ; sinon retourne tel quel"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cnts,_ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return roi
    c     = max(cnts, key=cv2.contourArea)
    peri  = cv2.arcLength(c, True)
    approx= cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx)==4:
        pts  = np.array([p[0] for p in approx], dtype="float32")
        rect = order_points(pts)
        (tl,tr,br,bl) = rect
        wA,wB = np.linalg.norm(br-bl), np.linalg.norm(tr-tl)
        hA,hB = np.linalg.norm(tr-br), np.linalg.norm(tl-bl)
        maxW, maxH = int(max(wA,wB)), int(max(hA,hB))
        dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
        M   = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(roi, M, (maxW, maxH))
    return roi

def enhance_and_binarise(img_bgr: np.ndarray) -> np.ndarray:
    """contraste, suppression ombre, N/B"""
    lab    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l,a,b  = cv2.split(lab)
    clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2     = clahe.apply(l)
    merged = cv2.merge((l2,a,b))
    enh    = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(31,31))
    bg     = cv2.morphologyEx(enh, cv2.MORPH_CLOSE, kernel)
    div    = cv2.divide(enh, bg, scale=255)

    gray   = cv2.cvtColor(div, cv2.COLOR_BGR2GRAY)
    bw     = cv2.adaptiveThreshold(gray,255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,15,2)
    return bw

def process_ecg(img_bgr: np.ndarray) -> tuple[np.ndarray, dict|None]:
    """pipeline complet → image N/B + bbox éventuelle"""
    bbox = detect_grid_openai(img_bgr)
    roi  = img_bgr[bbox["y"]:bbox["y"]+bbox["h"],
                   bbox["x"]:bbox["x"]+bbox["w"]] if bbox else hsv_fallback(img_bgr)
    roi  = perspective_or_bbox(roi)
    bw   = enhance_and_binarise(roi)
    return bw, bbox

# ───────────────────  ROUTES HTTP  ───────────────────
@app.route("/", methods=["GET"])
def home():
    return "<h3>API ECG (OpenAI Vision) opérationnelle.</h3>"

@app.route("/process", methods=["POST"])
def process_route():
    if "image" not in request.files:
        return jsonify({"error":"champ 'image' manquant"}), 400
    img_data = request.files["image"].read()
    nparr    = np.frombuffer(img_data, np.uint8)
    img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error":"image non décodable"}), 400

    bw, bbox = process_ecg(img_bgr)
    ok, buf  = cv2.imencode(".png", bw)
    if not ok:
        return jsonify({"error":"encodage PNG raté"}), 500
    bio = BytesIO(buf.tobytes())
    resp = send_file(bio, mimetype="image/png", as_attachment=False)
    # ajout de métadonnées bbox si dispo
    if bbox:
        resp.headers["X-Grid-Bbox"] = ",".join(str(v) for v in bbox.values())
    return resp

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print("🩺  ECG API lancée sur le port", port)
    app.run(host="0.0.0.0", port=port, debug=False)