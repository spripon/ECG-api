"""
ECG-API  –  extraction quadrillage + filtre “argenté”
© spripon – mai 2025
"""

import os, base64, json, cv2, numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI

# ───────── CONFIGURATION ─────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VISION_MODEL   = os.getenv("MODEL_NAME", "gpt-4o")
MAX_DIM        = 1600                                # redimension max pour Vision
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
CORS(app)

# ╭──────────────── warp v3 – Hough ─────────────╮
def warp(img, poly):
    """
    1. masque = polygone rose       → limite la recherche
    2. Canny + HoughLines           → gardes 2 lignes horizontales & 2 verticales
    3. intersections = 4 coins      → perspective parfaite
    4. si Hough échoue ⇒ warp_old()
    """
    # masque binaire
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask,
                 [np.array([[p["x"], p["y"]] for p in poly], dtype=int)],
                 255)

    # Canny + masque
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Hough
    lines = cv2.HoughLines(edges, 1, np.pi/180, 300)
    if lines is None or len(lines) < 4:
        return warp_old(img, poly)

    horiz, vert = [], []
    for r,t in lines[:,0]:
        ang = t*180/np.pi
        if abs(ang-90) < 10:  horiz.append((r,t))
        if abs(ang)   < 10:  vert .append((r,t))

    if len(horiz) < 2 or len(vert) < 2:
        return warp_old(img, poly)

    # extrêmes de chaque famille (rho min / rho max)
    def extremes(arr):
        arr = sorted(arr, key=lambda x: x[0])
        return arr[0], arr[-1]

    (rH1,tH1),(rH2,tH2) = extremes(horiz)
    (rV1,tV1),(rV2,tV2) = extremes(vert)

    # calcul intersections
    def intersect(r1,t1,r2,t2):
        A = np.array([[np.cos(t1), np.sin(t1)],
                      [np.cos(t2), np.sin(t2)]])
        b = np.array([r1, r2])
        x,y = np.linalg.solve(A, b)
        return [float(x), float(y)]

    TL = intersect(rH1,tH1, rV1,tV1)
    TR = intersect(rH1,tH1, rV2,tV2)
    BR = intersect(rH2,tH2, rV2,tV2)
    BL = intersect(rH2,tH2, rV1,tV1)
    rect = np.array([TL,TR,BR,BL], dtype=np.float32)

    # dimensions
    tl,tr,br,bl = rect
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))

    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (W,H))

# ---------- warp_old (fallback minAreaRect) ----------
def warp_old(img, poly):
    pts = np.array([[p["x"],p["y"]] for p in poly], dtype=np.float32)
    if pts.shape[0] != 4:
        pts = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32)
    rect = np.zeros((4,2), dtype=np.float32)
    s,d  = pts.sum(1), np.diff(pts,1)
    rect[0],rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1],rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    tl,tr,br,bl = rect
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (W,H))

# ╭──────────────── Coupe bande blanche ─────────────╮
def crop_white_top(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1].astype(np.float32)
    idx = np.argmax(sat.mean(axis=1) > 25)           # S>25 ≈ début rose
    return img[idx+5:,:] if idx>0 else img, idx+5

# ╭──────────────── Détection couleur robuste ───────╮
def detect_color(img):
    roi, offset = crop_white_top(img)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    ab  = lab[:,:,1:3].reshape(-1,2).astype(np.float32)
    _, lbl, ctr = cv2.kmeans(
        ab, 3, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0),
        10, cv2.KMEANS_PP_CENTERS)
    ros_idx = np.argmax(ctr[:,0])
    mask = (lbl.reshape(lab.shape[:2]) == ros_idx).astype(np.uint8)*255

    # supprime les zones trop peu saturées (blanc/ombre)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask[hsv[:,:,1] < 40] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((25,25), np.uint8))
    mask = cv2.dilate(mask, np.ones((15,15), np.uint8), 2)

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # garde contours ≥30 % du plus grand
    main = max(cnts, key=cv2.contourArea)
    Amax = cv2.contourArea(main)
    good = [c for c in cnts if cv2.contourArea(c) > 0.3*Amax]
    c    = max(good, key=cv2.contourArea)

    pts = cv2.boxPoints(cv2.minAreaRect(c)).astype(int)
    pts[:,1] += offset

    xs,ys = pts[:,0], pts[:,1]
    M = int(0.05*max(xs.ptp(), ys.ptp()))              # marge 5 %
    x,y   = int(xs.min()-M), int(ys.min()-M)
    w,h   = int(xs.ptp()+2*M), int(ys.ptp()+2*M)

    poly = [{"x":x,     "y":y},
            {"x":x+w-1, "y":y},
            {"x":x+w-1, "y":y+h-1},
            {"x":x,     "y":y+h-1}]
    return {"x":x,"y":y,"w":w,"h":h,"poly":poly}

# ╭──────────────── GPT-4o Vision fallback ──────────╮
def detect_vision(img):
    if client is None:
        return None
    h,w = img.shape[:2]; scale = 1
    if max(h,w) > MAX_DIM:
        scale = MAX_DIM/max(h,w)
        img_r = cv2.resize(img,(int(w*scale), int(h*scale)))
    else:
        img_r = img
    _,buf = cv2.imencode(".png", img_r)
    b64   = base64.b64encode(buf).decode()

    messages=[{"role":"user","content":[
        {"type":"text","text":"Isoler la zone quadrillée rose de l'ECG. "
                              "Retourne 4-8 coins JSON."},
        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]}]
    tools=[{"type":"function","function":{"name":"set","parameters":{
        "type":"object","properties":{
            "points":{"type":"array","items":{
                "type":"object","properties":{
                    "x":{"type":"integer"},"y":{"type":"integer"}},
                "required":["x","y"]},
            "minItems":4,"maxItems":8}},
        "required":["points"]}}}]

    try:
        r = client.chat.completions.create(
                model=VISION_MODEL, messages=messages,
                tools=tools,
                tool_choice={"type":"function","function":{"name":"set"}},
                temperature=0)
        args = r.choices[0].message.tool_calls[0].function.arguments
        if isinstance(args,str): args=json.loads(args)
        pts = args["points"]
        return {"poly":[{"x":p["x"]/scale,"y":p["y"]/scale} for p in pts]}
    except Exception as e:
        print("Vision error:", e)
        return None

# ╭──────────────── Filtre “argenté” ───────────────╮
def enhance(img):
    exp = cv2.convertScaleAbs(img, alpha=1.2, beta=20)     # +20 % expo
    hsv = cv2.cvtColor(exp, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = (hsv[:,:,1]*0.5).astype(np.uint8)         # désaturation 50 %
    exp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bg  = cv2.medianBlur(exp, 51)                          # supprime ombres
    flat= cv2.divide(exp, bg, scale=255)
    return cv2.addWeighted(flat, 1.3,
                           cv2.GaussianBlur(flat,(0,0),3),
                           -0.3, 0)

# ╭──────────────── Pipeline ───────────────────────╮
def process(img):
    if img.shape[0] > img.shape[1]:                       # portrait → paysage
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    box = detect_color(img) or detect_vision(img)
    if not box:
        raise ValueError("zone non trouvée")

    roi = warp(img, box["poly"])
    return enhance(roi)

# ╭──────────────── Flask routes ───────────────────╮
@app.route("/process", methods=["POST"])
def handle():
    if "image" not in request.files:
        return jsonify(error="image manquante"),400
    img = cv2.imdecode(np.frombuffer(request.files["image"].read(), np.uint8),
                       cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="decode"),400
    try:
        out = process(img)
        _,buf = cv2.imencode(".png", out)
        return send_file(BytesIO(buf.tobytes()), mimetype="image/png")
    except Exception as e:
        return jsonify(error=str(e)),500

@app.route("/")
def home():
    return "<h3>ECG-API – quadrillage + filtre argenté</h3>"

if __name__ == "__main__":
    app.run(host="0.0.0.0",
            port=int(os.getenv("PORT", 8080)),
            debug=False)