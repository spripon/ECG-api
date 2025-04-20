"""
ECG‑API  – GPT‑4o Vision ✚ Segmentation couleur ✚ FFT
© spripon – mai 2025
"""

import os, base64, json, cv2, numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI

# ╭──── CONFIG ────╮
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL          = os.getenv("MODEL_NAME", "gpt-4o")     # gpt‑4o ou gpt‑4o‑mini
MAX_DIM        = 1600                                  # redim Vision
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
# ╰────────────────╯

app = Flask(__name__)
CORS(app)

# ─────────────────── utilitaires ───────────────────
def order_pts(pts):
    rect = np.zeros((4,2), dtype="float32")
    s, d = pts.sum(1), np.diff(pts, axis=1)
    rect[0],rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1],rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

def warp(roi, poly):
    # poly : liste [{"x":..,"y":..}, …]  --->  Nx2
    pts = np.array([[p["x"], p["y"]] for p in poly], dtype="float32")
    if len(pts) < 4:
        return roi
    if len(pts) > 4:
        pts = cv2.convexHull(pts)
    if len(pts) != 4:                      # rectangle min
        pts = cv2.boxPoints(cv2.minAreaRect(pts)).astype("float32")

    rect = order_pts(pts)
    tl,tr,br,bl = rect
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M   = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(roi, M, (W, H))

# ───────── Vision  ─────────
def vision_bbox(img):
    if client is None: return None
    h,w = img.shape[:2]; scale = 1.0
    if max(h,w) > MAX_DIM:
        scale = MAX_DIM/max(h,w)
        img_r = cv2.resize(img,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)
    else: img_r = img

    _,buf = cv2.imencode(".png", img_r)
    b64   = base64.b64encode(buf).decode()

    messages=[{"role":"user","content":[
      {"type":"text","text":(
        "Localise uniquement la zone quadrillée ECG (teinte rose/rouge pâle/"
        "orange pâle/jaune pâle). Ignore la bande blanche supérieure, bords "
        "latéraux, trous de classeur, coins foncés. "
        "Réponds JSON {\"points\":[{\"x\":int,\"y\":int}]} (4‑8 coins).")},
      {"type":"image_url",
       "image_url":{"url":f"data:image/png;base64,{b64}"}}]}]

    tools=[{"type":"function","function":{"name":"set",
        "parameters":{"type":"object","properties":{
          "points":{"type":"array","items":{
              "type":"object","properties":{
                  "x":{"type":"integer"},"y":{"type":"integer"}},
              "required":["x","y"]},
              "minItems":4,"maxItems":8}},
        "required":["points"]}}}]

    try:
        rsp=client.chat.completions.create(model=MODEL,
                messages=messages,tools=tools,
                tool_choice={"type":"function","function":{"name":"set"}},
                temperature=0)
        pts=rsp.choices[0].message.tool_calls[0].function.arguments
        if isinstance(pts,str): pts=json.loads(pts)
        pts=pts["points"]
        xs,ys=[p["x"] for p in pts],[p["y"] for p in pts]
        x,y = min(xs),min(ys); w2,h2 = max(xs)-x, max(ys)-y
        return {"x":int(x/scale),"y":int(y/scale),
                "w":int(w2/scale),"h":int(h2/scale),
                "poly":[{"x":int(p["x"]/scale),"y":int(p["y"]/scale)} for p in pts]}
    except Exception as e:
        print("Vision error:", e)
        return None

def vision_ok(img, box, thr=0.40):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    l1,u1 = np.array([0,25,40]), np.array([22,255,255])
    l2,u2 = np.array([158,25,40]),np.array([179,255,255])
    mask=cv2.inRange(hsv,l1,u1)|cv2.inRange(hsv,l2,u2)
    x,y,w,h = box["x"],box["y"],box["w"],box["h"]
    ratio   = mask[y:y+h, x:x+w].sum()/255/(w*h)
    return ratio >= thr

# ───────── HSV fallback ─────────
def detect_color(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    l1,u1 = np.array([0,25,40]), np.array([22,255,255])
    l2,u2 = np.array([158,25,40]),np.array([179,255,255])
    mask=cv2.inRange(hsv,l1,u1)|cv2.inRange(hsv,l2,u2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c=max(cnts,key=cv2.contourArea)
    rect=cv2.minAreaRect(c); box=cv2.boxPoints(rect).astype(int)
    xs,ys = box[:,0], box[:,1]
    return {"x":xs.min(),"y":ys.min(),"w":xs.ptp(),"h":ys.ptp(),
            "poly":[{"x":int(x),"y":int(y)} for x,y in box]}

# ───────── FFT fallback ─────────
def grid_ratio(crop):
    g=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    return (g<200).sum()/g.size

def fft_flood(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mag=np.log(np.abs(np.fft.fftshift(np.fft.fft2(g)))+1)
    y0=np.argmax(cv2.GaussianBlur(mag.mean(1),(51,1),0))
    mask=np.zeros_like(g); mask[y0,:]=255
    seed=(g.shape[1]//2,y0)
    mask=cv2.floodFill(cv2.cvtColor(g,cv2.COLOR_GRAY2BGR),
                       None,seed,255,loDiff=(5,5,5),upDiff=(5,5,5))[1][:,:,0]
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h=cv2.boundingRect(max(cnts,key=cv2.contourArea))
    return {"x":x,"y":y,"w":w,"h":h,
            "poly":[{"x":x,"y":y},{"x":x+w,"y":y},
                     {"x":x+w,"y":y+h},{"x":x,"y":y+h}]}

# ───────── masque HSV pour affiner bbox ─────────
def hsv_mask(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    l1,u1 = np.array([0,25,40]), np.array([22,255,255])
    l2,u2 = np.array([158,25,40]),np.array([179,255,255])
    return cv2.inRange(hsv,l1,u1)|cv2.inRange(hsv,l2,u2)

def refine_bbox(img, box, thr=0.05):
    x,y,w,h = box["x"],box["y"],box["w"],box["h"]
    crop = hsv_mask(img[y:y+h, x:x+w])
    rows = crop.sum(1)/255; cols = crop.sum(0)/255; H,W = crop.shape
    top = next(i for i,v in enumerate(rows) if v > thr*W)
    bottom = H-1-next(i for i,v in enumerate(rows[::-1]) if v > thr*W)
    left = next(i for i,v in enumerate(cols) if v > thr*H)
    right = W-1-next(i for i,v in enumerate(cols[::-1]) if v > thr*H)
    box.update({"x":x+left,"y":y+top,"w":right-left+1,"h":bottom-top+1,
                "poly":[{"x":x+left,"y":y+top},
                        {"x":x+right,"y":y+top},
                        {"x":x+right,"y":y+bottom},
                        {"x":x+left,"y":y+bottom}]})
    return box

# ───────── amélioration NB ─────────
def enhance(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab); l=cv2.createCLAHE(2.0,(16,16)).apply(l)
    img=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
    norm=cv2.divide(img,cv2.blur(img,(101,101)),scale=255)
    gray=cv2.cvtColor(norm,cv2.COLOR_BGR2GRAY)
    bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                             cv2.THRESH_BINARY,35,7)
    if np.mean(bw[:30,:30])<128: bw=cv2.bitwise_not(bw)
    return bw

# ───────── pipeline ─────────
def process(img):
    # rotation portrait -> paysage
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    box = vision_bbox(img)
    if not box or not vision_ok(img, box):
        box = detect_color(img)
        if grid_ratio(img[box["y"]:box["y"]+box["h"],
                         box["x"]:box["x"]+box["w"]]) < 0.10:
            box = fft_flood(img)

    # Affiner la bbox avec masque couleur pour retirer bande blanche
    box = refine_bbox(img, box, thr=0.05)

    roi = img[box["y"]:box["y"]+box["h"], box["x"]:box["x"]+box["w"]]
    roi = warp(roi, box["poly"])
    return enhance(roi)

# ───────── Flask routes ─────────
@app.route("/process", methods=["POST"])
def handle():
    if "image" not in request.files:
        return jsonify(error="image manquante"),400
    data=request.files["image"].read()
    img = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="decode"),400
    out = process(img)
    _, buf = cv2.imencode(".png", out)
    return send_file(BytesIO(buf.tobytes()), mimetype="image/png")

@app.route("/")
def home(): return "<h3>ECG‑API — Vision + HSV + FFT (bande blanche supprimée)</h3>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",8080)), debug=False)