"""
ECG API – Flask + OpenCV + OpenAI Vision (auto‑compatible)
"""

import os, base64, cv2, numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI

# ───────────────── CONFIG ─────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PREFERRED_MODEL = os.getenv("MODEL_NAME", "gpt-4o")
MAX_DIM_VISION  = int(os.getenv("VISION_MAX_DIM", "1600"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
CANDIDATE_MODELS = [
    PREFERRED_MODEL, "gpt-4o", "gpt-4o-mini",
    "gpt-4-turbo", "gpt-4o-preview", "gpt-4-vision-preview"
]

def pick_first_available_model():
    if client is None:
        return None
    try:
        remote = {m.id for m in client.models.list().data}
    except Exception:
        remote = set()
    for m in CANDIDATE_MODELS:
        if m in remote:
            return m
    for m in CANDIDATE_MODELS:           # ping 1 token
        try:
            client.chat.completions.create(
                model=m,
                messages=[{"role":"user","content":[{"type":"text","text":"ping"}]}],
                max_tokens=1
            )
            return m
        except Exception:
            continue
    return None

VISION_MODEL = pick_first_available_model()
print(("✅ Modèle Vision : " + VISION_MODEL) if VISION_MODEL
      else "⚠️ Aucun modèle Vision, fallback HSV")

# ─────────────── FLASK APP ───────────────
app = Flask(__name__)
CORS(app)

# ---------- utilitaire ----------
def order_points(pts):
    rect = np.zeros((4,2),dtype="float32")
    s, d = pts.sum(1), np.diff(pts,axis=1)
    rect[0],rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1],rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

# ---------- Vision ----------
def detect_grid_openai(img):
    if VISION_MODEL is None:
        return None
    h0,w0 = img.shape[:2]
    scale = 1.0
    if max(h0,w0) > MAX_DIM_VISION:
        scale = MAX_DIM_VISION / max(h0,w0)
        img_r = cv2.resize(img, (int(w0*scale), int(h0*scale)))
    else:
        img_r = img
    ok,buf = cv2.imencode(".png", img_r);  b64 = base64.b64encode(buf).decode()
    messages=[{
        "role":"user",
        "content":[
            {"type":"text",
             "text":"Localise la zone quadrillée ECG (D1‑D3, aVR‑aVF, V1‑V6) et renvoie JSON {x,y,w,h}."},
            {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}
        ]}]
    tools=[{
        "type":"function",
        "function":{
            "name":"set_bbox",
            "parameters":{
                "type":"object",
                "properties":{
                    "x":{"type":"integer"},
                    "y":{"type":"integer"},
                    "w":{"type":"integer"},
                    "h":{"type":"integer"}},
                "required":["x","y","w","h"]}}}]
    try:
        rsp=client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            tools=tools,
            tool_choice={"type":"function","function":{"name":"set_bbox"}},  # ← correct
            temperature=0)
        bbox=rsp.choices[0].message.tool_calls[0].arguments
        for k in bbox: bbox[k]=int(bbox[k]/scale)
        return bbox
    except Exception as e:
        print("Vision API error:",e)
        return None

# ---------- Fallback HSV ----------
def hsv_fallback(img):
    blur=cv2.GaussianBlur(img,(5,5),0)
    hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    sat,val=20,90
    ranges=[([0,sat,val],[15,255,255]),([165,sat,val],[180,255,255]),
            ([16,sat,val],[45,255,255])]
    mask=None
    for lo,hi in ranges:
        cur=cv2.inRange(hsv,np.array(lo),np.array(hi))
        mask=cur if mask is None else cv2.bitwise_or(mask,cur)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8),3)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return img
    x,y,w,h=cv2.boundingRect(max(cnts,key=cv2.contourArea))
    return img[y:y+h,x:x+w]

# ---------- Warp ----------
def perspective_or_bbox(roi):
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    cnts,_=cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return roi
    c=max(cnts,key=cv2.contourArea)
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*peri,True)
    if len(approx)==4:
        rect=order_points(np.array([p[0] for p in approx]))
        (tl,tr,br,bl)=rect
        w=int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl)))
        h=int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
        dst=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype="float32")
        M=cv2.getPerspectiveTransform(rect,dst)
        return cv2.warpPerspective(roi,M,(w,h))
    return roi

# ---------- Enhance + binarise ----------
def enhance_and_binarise(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(3.0,(8,8)).apply(l)
    enh=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
    bg=cv2.morphologyEx(enh,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(31,31)))
    div=cv2.divide(enh,bg,scale=255)
    gray=cv2.cvtColor(div,cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,15,2)

def process_ecg(img):
    bbox=detect_grid_openai(img)
    roi=img[bbox["y"]:bbox["y"]+bbox["h"],bbox["x"]:bbox["x"]+bbox["w"]] if bbox else hsv_fallback(img)
    roi=perspective_or_bbox(roi)
    bw=enhance_and_binarise(roi)
    return bw,bbox

# ─────────────── Routes ──────────────
@app.route("/",methods=["GET"])
def home(): return "<h3>ECG API opérationnelle.</h3>"

@app.route("/process",methods=["POST"])
def proc():
    if 'image' not in request.files:
        return jsonify(error="champ 'image' manquant"),400
    data=request.files['image'].read()
    img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="image illisible"),400
    bw,bbox=process_ecg(img)
    ok,buf=cv2.imencode(".png",bw)
    if not ok: return jsonify(error="encodage PNG raté"),500
    bio=BytesIO(buf.tobytes())
    resp=send_file(bio,mimetype="image/png")
    if bbox: resp.headers["X-Grid-Bbox"]=",".join(map(str,bbox.values()))
    return resp

if __name__=="__main__":
    port=int(os.getenv("PORT","8080"))
    app.run(host="0.0.0.0",port=port,debug=False)
