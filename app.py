import os, base64, json, cv2, numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VISION_MODEL   = os.getenv("MODEL_NAME", "gpt-4-vision-preview")
MAX_DIM    = 1600
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
CORS(app)

def preprocess_image(img):
    # Améliorer le contraste initial pour mieux détecter les zones
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Réduction légère du bruit
    img = cv2.GaussianBlur(img, (3,3), 0)
    return img

def order(pts):
    rect = np.zeros((4,2),dtype="float32")
    s,d  = pts.sum(1), np.diff(pts,1)
    rect[0],rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1],rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

def detect_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Définition des plages de couleurs (rose pâle, rouge pâle, jaune pâle, orange pâle)
    masks = []

    # Rose pâle à rouge pâle
    masks.append(cv2.inRange(hsv, np.array([150, 20, 180]), np.array([179, 110, 255])))
    masks.append(cv2.inRange(hsv, np.array([0, 20, 180]), np.array([20, 110, 255])))

    # Jaune pâle à orange pâle
    masks.append(cv2.inRange(hsv, np.array([20, 20, 180]), np.array([45, 110, 255])))

    # Combinaison des masques
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)

    # Amélioration du masque
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Détection des contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Sélection du plus grand contour
    cnt = max(cnts, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Si on n'a pas exactement 4 points, utiliser minAreaRect
    if len(approx) != 4:
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect)
    else:
        pts = approx.reshape(-1, 2)

    pts = pts.astype(int)
    xs, ys = pts[:,0], pts[:,1]
    return {
        "poly": [{"x": int(x), "y": int(y)} for x, y in pts],
        "x": int(xs.min()),
        "y": int(ys.min()),
        "w": int(np.ptp(xs)),
        "h": int(np.ptp(ys))
    }

def detect_vision(img):
    if client is None: return None
    h,w = img.shape[:2]; scale=1
    if max(h,w)>MAX_DIM:
        scale=MAX_DIM/max(h,w)
        img=cv2.resize(img,(int(w*scale),int(h*scale)))
    _,buf=cv2.imencode(".png",img)
    b64=base64.b64encode(buf).decode()
    msg=[{"role":"user","content":[
        {"type":"text","text":"Donne 4-8 points de la zone quadrillée ECG."},
        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]}]
    tools=[{"type":"function","function":{"name":"set","parameters":{
        "type":"object","properties":{
            "points":{"type":"array","items":{
                "type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},
                "required":["x","y"]},"minItems":4,"maxItems":8}},
            "required":["points"]}}}]
    r=client.chat.completions.create(model=VISION_MODEL,messages=msg,tools=tools,
        tool_choice={"type":"function","function":{"name":"set"}},temperature=0)
    args=r.choices[0].message.tool_calls[0].function.arguments
    if isinstance(args,str): args=json.loads(args)
    pts=args["points"]
    return {"poly":[{"x":p["x"]/scale,"y":p["y"]/scale} for p in pts]}

def warp(img, poly):
    pts = np.array([[p["x"], p["y"]] for p in poly], dtype="float32")
    if pts.shape[0] != 4:
        pts = cv2.boxPoints(cv2.minAreaRect(pts)).astype("float32")

    # Ordonner les points
    rect = order(pts)
    (tl, tr, br, bl) = rect

    # Calculer les dimensions maximales
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Points de destination
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Transformation perspective
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # Correction de l'angle basée sur la détection des lignes
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    if lines is not None:
        angles = []
        for rho, theta in lines[0]:
            angle = theta * 180 / np.pi
            if angle < 45:
                angles.append(angle)
            elif angle > 135:
                angles.append(angle - 180)

        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:  # Ne corriger que si l'angle est significatif
                center = (warped.shape[1] // 2, warped.shape[0] // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                warped = cv2.warpAffine(warped, M, (warped.shape[1], warped.shape[0]))

    return warped

def enhance(img):
    # exposition + brillance
    exp = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
    # désaturation
    hsv = cv2.cvtColor(exp, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = (hsv[:,:,1]*0.5).astype(np.uint8)
    exp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # suppression des ombres
    bg = cv2.medianBlur(exp, 51)
    flat = cv2.divide(exp, bg, scale=255)
    # sharpen
    sh = cv2.addWeighted(flat, 1.3, cv2.GaussianBlur(flat,(0,0),3), -0.3, 0)
    return sh

def process(img):
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Prétraitement
    img = preprocess_image(img)

    box = detect_color(img) or detect_vision(img)
    if not box:
        raise ValueError("zone non trouvée")

    roi = warp(img, box["poly"])
    return enhance(roi)

@app.route("/process", methods=["POST"])
def handle():
    if "image" not in request.files:
        return jsonify(error="image?"), 400
    img = cv2.imdecode(np.frombuffer(request.files["image"].read(), np.uint8),
        cv2.IMREAD_COLOR)
    try:
        out = process(img)
        _, buf = cv2.imencode(".png", out)
        return send_file(BytesIO(buf.tobytes()), mimetype="image/png")
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/")
def home():
    return "<h3>ECG-API filtre argenté</h3>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)
