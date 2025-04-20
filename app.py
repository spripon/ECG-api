"""
ECG‑API – GPT‑4o Vision ✚ HSV ✚ FFT
© spripon – mai 2025
"""

import os, base64, json, cv2, numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI

# ───── CONFIG ─────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL          = os.getenv("MODEL_NAME", "gpt-4o")     # gpt‑4o, gpt‑4o‑mini…
MAX_DIM        = 1600                                  # redimension Vision
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
CORS(app)

# ───── utilitaires ─────
def order_pts(pts):
    rect = np.zeros((4,2),dtype="float32")
    s, d = pts.sum(1), np.diff(pts, axis=1)
    rect[0],rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1],rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

def warp(roi, poly):
    # poly : liste de {"x":..,"y":..} → tableau Nx2
    pts = np.array([[p["x"], p["y"]] for p in poly], dtype="float32")
    if len(pts) < 4:
        return roi
    if len(pts) > 4:
        pts = cv2.convexHull(pts)
    if len(pts) != 4:
        pts = cv2.boxPoints(cv2.minAreaRect(pts)).astype("float32")

    rect = order_pts(pts)
    tl,tr,br,bl = rect
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(roi, M, (W,H))

# ───── 1. Vision ─────
def vision_bbox(img):
    if client is None: return None
    h,w = img.shape[:2]; scale = 1.0
    if max(h,w) > MAX_DIM:
        scale = MAX_DIM/max(h,w)
        img_r = cv2.resize(img,(int(w*scale),int(h*scale)))
    else: img_r = img

    ok,buf = cv2.imencode(".png", img_r)
    b64 = base64.b64encode(buf).decode()
    messages=[{"role":"user","content":[
      {"type":"text","text":(
        "Repère UNIQUEMENT la zone quadrillée ECG (rose, rouge pâle, orange pâle, "
        "jaune pâle). Ignore bande blanche supérieure, marges latérales, bas foncé, "
        "classeurs. Réponds JSON {\"points\":[{\"x\":int,\"y\":int}]} 4‑8 sommets.")},
      {"type":"image_url",
       "image_url":{"url":f"data:image/png;base64,{b64}"}}]}]
    tools=[{"type":"function","function":{"name":"set",
        "parameters":{"type":"object","properties":{
          "points":{"type":"array","items":{
              "type":"object",
              "properties":{"x":{"type":"integer"},"y":{"type":"integer"}},
              "required":["x","y"]},
              "minItems":4,"maxItems":8}},
        "required":["points"]}}}]

    try:
        rsp=client.chat.completions.create(
            model=MODEL,messages=messages,tools=tools,
            tool_choice={"type":"function","function":{"name":"set"}},
            temperature=0)
        pts = rsp.choices[0].message.tool_calls[0].function.arguments
        if isinstance(pts,str):
            pts = json.loads(pts)
        pts = pts["points"]
        xs,ys = [p["x"] for p in pts],[p["y"] for p in pts]
        x,y = min(xs),min(ys); w2,h2 = max(xs)-x, max(ys)-y
        return {"x":int(x/scale),"y":int(y/scale),
                "w":int(w2/scale),"h":int(h2/scale),
                "poly":[{"x":int(p["x"]/scale),"y":int(p["y"]/scale)} for p in pts]}
    except Exception as e:
        print("Vision err:", e)
        return None

def vision_ok(img, box, thr=0.40):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    l1,u1 = np.array([0,30,50]), np.array([20,255,255])
    l2,u2 = np.array([160,30,50]),np.array([179,255,255])
    mask=cv2.inRange(hsv,l1,u1)|cv2.inRange(hsv,l2,u2)
    x,y,w,h = box["x"],box["y"],box["w"],box["h"]
    ratio = mask[y:y+h, x:x+w].sum()/255/(w*h)
    return ratio >= thr

# ───── 2. Fallback couleur HSV ─────
def detect_color(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    l1,u1 = np.array([0,30,50]), np.array([20,255,255])
    l2,u2 = np.array([160,30,50]),np.array([179,255,255])
    mask=cv2.inRange(hsv,l1,u1)|cv2.inRange(hsv,l2,u2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c=max(cnts,key=cv2.contourArea)
    rect=cv2.minAreaRect(c); box=cv2.boxPoints(rect).astype(int)
    xs,ys = box[:,0], box[:,1]
    return {"x":xs.min(),"y":ys.min(),"w":xs.ptp(),"h":ys.ptp(),
            "poly":[{"x":int(x),"y":int(y)} for x,y in box]}

# ───── 3. Fallback FFT ─────
def grid_ratio(crop):
    g=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    return (g<200).sum()/g.size

def fft_flood(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mag=np.log(np.abs(np.fft.fftshift(np.fft.fft2(g)))+1)
    peak=np.argmax(cv2.GaussianBlur(mag.mean(1),(51,1),0))
    mask=np.zeros_like(g); mask[peak,:]=255
    seed=(g.shape[1]//2, peak)
    mask=cv2.floodFill(cv2.cvtColor(g,cv2.COLOR_GRAY2BGR),
                       None,seed,255,loDiff=(5,5,5),upDiff=(5,5,5))[1][:,:,0]
    x,y,w,h=cv2.boundingRect(max(cv2.findContours(mask,cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea))
    return {"x":x,"y":y,"w":w,"h":h,
            "poly":[{"x":x,"y":y},{"x":x+w,"y":y},
                    {"x":x+w,"y":y+h},{"x":x,"y":y+h}]}

# ───── amélioration NB ─────
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

# ───── pipeline ─────
def process(img):
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)

    box = vision_bbox(img)
    if not box or not vision_ok(img, box):
        box = detect_color(img)
        if grid_ratio(img[box["y"]:box["y"]+box["h"],
                         box["x"]:box["x"]+box["w"]]) < 0.10:
            box = fft_flood(img)

    roi = img[box["y"]:box["y"]+box["h"], box["x"]:box["x"]+box["w"]]
    roi = warp(roi, box["poly"])
    return enhance(roi)

# ───── routes HTTP ─────
@app.route("/process", methods=["POST"])
def proc():
    if "image" not in request.files:
        return jsonify(err="image manquante"),400
    buf=request.files["image"].read()
    img=cv2.imdecode(np.frombuffer(buf,np.uint8),cv2.IMREAD_COLOR)
    if img is None: return jsonify(err="decode"),400
    out=process(img);_,png=cv2.imencode(".png",out)
    return send_file(BytesIO(png.tobytes()),mimetype="image/png")

@app.route("/")
def index(): return "<h3>ECG‑API – Vision + HSV + FFT</h3>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",8080)), debug=False)
