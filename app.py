"""
ECG‑API – Flask + OpenCV + GPT‑4o Vision
© spripon – mai 2025
"""

# ───────────── imports ─────────────
import os, base64, json, cv2, numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI

# ───────────── config ──────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VISION_MODEL   = os.getenv("MODEL_NAME", "gpt-4o")   # gpt‑4o / gpt‑4o‑mini …

MAX_DIM_VISION = 1600                                # ≤ 1600 px → coût min
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
CORS(app)

# ───────── utilitaires cv2 ─────────
def order_pts(pts):
    rect = np.zeros((4,2),dtype="float32")
    s, d = pts.sum(1), np.diff(pts,axis=1)
    rect[0],rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1],rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

# ─────── crop bande blanche FFT ────
def crop_to_grid(gray):
    h,w = gray.shape
    f   = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log(np.abs(f)+1)
    proj= cv2.GaussianBlur(mag.mean(1),(51,1),0)
    peak= int(np.argmax(proj))
    top = max(0, peak - int(0.45*h))
    bot = min(h, peak + int(0.45*h))
    return gray[top:bot]

# ───── Vision → bbox quadrillage ───
def detect_grid(img):
    if client is None: return None
    h0,w0 = img.shape[:2]
    scale = 1.0
    if max(h0,w0) > MAX_DIM_VISION:
        scale = MAX_DIM_VISION / max(h0,w0)
        img_r = cv2.resize(img,(int(w0*scale),int(h0*scale)))
    else: img_r = img
    ok,buf=cv2.imencode(".png",img_r); b64=base64.b64encode(buf).decode()

    messages=[{
        "role":"user",
        "content":[
            {"type":"text",
             "text":(
               "Délimite UNIQUEMENT la zone quadrillée rose/orange contenant les 12 "
               "dérivations ECG (D1‑D3, aVR‑aVF, V1‑V6). "
               "Ignore bande blanche du haut, marges latérales, bas sans quadrillage. "
               "Renvoie JSON : {\"points\":[{\"x\":int,\"y\":int},…]} 4‑8 sommets.")},
            {"type":"image_url",
             "image_url":{"url":f"data:image/png;base64,{b64}"}}]}]

    tools=[{
        "type":"function",
        "function":{
            "name":"set_grid",
            "parameters":{
              "type":"object",
              "properties":{"points":{"type":"array",
                                      "items":{"type":"object",
                                               "properties":{"x":{"type":"integer"},
                                                             "y":{"type":"integer"}},
                                               "required":["x","y"]},
                                      "minItems":4,"maxItems":8}},
              "required":["points"]}}}]

    rsp=client.chat.completions.create(
        model=VISION_MODEL,messages=messages,tools=tools,
        tool_choice={"type":"function","function":{"name":"set_grid"}},
        temperature=0)
    pts=rsp.choices[0].message.tool_calls[0].function.arguments
    if isinstance(pts,str): pts=json.loads(pts)
    pts=pts["points"];        xs=[p["x"] for p in pts]; ys=[p["y"] for p in pts]
    x,y=min(xs),min(ys); w=max(xs)-x; h=max(ys)-y
    if w*h < 0.20*w0*h0 or w/h>2.2: return None
    return {"x":int(x/scale),"y":int(y/scale),
            "w":int(w/scale),"h":int(h/scale),"poly":[
            {"x":int(p["x"]/scale),"y":int(p["y"]/scale)} for p in pts]}

# ───── warp perspective ─────
def four_point_warp(roi, poly):
    if len(poly)<4: return roi
    rect=order_pts(np.array([[p["x"],p["y"]] for p in poly],dtype="float32"))
    (tl,tr,br,bl)=rect
    w=int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl)))
    h=int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
    dst=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype="float32")
    M=cv2.getPerspectiveTransform(rect,dst)
    return cv2.warpPerspective(roi,M,(w,h))

# ───── améliorations image ────
def enhance_bw(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(2.0,(16,16)).apply(l)
    img=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
    # suppression ombre
    bg=cv2.blur(img,(101,101))
    norm=cv2.divide(img,bg,scale=255)
    gray=cv2.cvtColor(norm,cv2.COLOR_BGR2GRAY)
    bw=cv2.adaptiveThreshold(gray,255,
                             cv2.ADAPTIVE_THRESH_MEAN_C,
                             cv2.THRESH_BINARY,35,8)
    if np.mean(bw[:30,:30])<128: bw=cv2.bitwise_not(bw)
    return crop_to_grid(bw)

# ───── pipeline complet ─────
def process(img):
    # rotation portrait → paysage
    if img.shape[0] > img.shape[1]:
        img=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)

    bbox=detect_grid(img)
    roi = (img[bbox["y"]:bbox["y"]+bbox["h"],
               bbox["x"]:bbox["x"]+bbox["w"]]
           if bbox else img)
    if bbox: roi=four_point_warp(roi,bbox["poly"])
    out=enhance_bw(roi)
    return out

# ───── Routes HTTP ─────
@app.route("/",methods=["GET"])
def index():
    return "<h3>ECG API Vision prête.</h3>"

@app.route("/process",methods=["POST"])
def proc():
    if "image" not in request.files:
        return jsonify(error="image manquante"),400
    data=request.files["image"].read()
    img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if img is None: return jsonify(error="image illisible"),400
    out=process(img)
    _,buf=cv2.imencode(".png",out)
    return send_file(BytesIO(buf.tobytes()),
                     mimetype="image/png")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",8080)),debug=False)
