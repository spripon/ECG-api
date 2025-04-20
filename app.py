"""
ECG‑API  – Vision + grid‑segmentation fallback
© spripon 05/2025
"""

import os, base64, json, cv2, numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI

# ───────────── CONFIG ─────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL          = os.getenv("MODEL_NAME", "gpt-4o")
MAX_DIM        = 1600
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
CORS(app)

# ───── utilitaires ─────
def order(p): s,d=p.sum(1),np.diff(p,1)
def order(p):
    r=np.zeros((4,2),dtype="float32");s,d=p.sum(1),np.diff(p,1)
    r[0],r[2]=p[np.argmin(s)],p[np.argmax(s)]
    r[1],r[3]=p[np.argmin(d)],p[np.argmax(d)];return r

# ───── Vision bbox ─────
def vision_bbox(img):
    if client is None: return None
    h,w=img.shape[:2];scale=1.0
    if max(h,w)>MAX_DIM:
        scale=MAX_DIM/max(h,w)
        img=cv2.resize(img,(int(w*scale),int(h*scale)))
    ok,buf=cv2.imencode(".png",img);b64=base64.b64encode(buf).decode()
    msg=[{"role":"user","content":[
        {"type":"text","text":(
          "Repère exclusivement la zone quadrillée rose/orange avec les 12 dérivations "
          "ECG. Ignore les marges et bande blanche. Réponds JSON "
          "{\"points\":[{\"x\":int,\"y\":int},…]}.")},
        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]}]
    tools=[{"type":"function","function":{"name":"set","parameters":{
        "type":"object","properties":{"points":{"type":"array","items":{
            "type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},
            "required":["x","y"]},"minItems":4,"maxItems":8}},
        "required":["points"]}}}]
    try:
        r=client.chat.completions.create(model=MODEL,messages=msg,
               tools=tools,tool_choice={"type":"function","function":{"name":"set"}},
               temperature=0)
        pts=r.choices[0].message.tool_calls[0].function.arguments
        if isinstance(pts,str):pts=json.loads(pts)
        pts=pts["points"];xs=[p["x"] for p in pts];ys=[p["y"] for p in pts]
        x,y=min(xs),min(ys);w2=max(xs)-x;h2=max(ys)-y
        return {"x":int(x/scale),"y":int(y/scale),
                "w":int(w2/scale),"h":int(h2/scale),
                "poly":[{"x":int(p["x"]/scale),"y":int(p["y"]/scale)} for p in pts]}
    except Exception as e:
        print("Vision err:",e);return None

# ───── validation grille ─────
def grid_ratio(crop):
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    lines=(gray<200).sum()/gray.size
    return lines  # ~0.25 si grille, <0.05 si bande blanche

# ───── fallback FFT + flood ─────
def fft_flood(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    f=np.fft.fftshift(np.fft.fft2(g));mag=np.log(np.abs(f)+1)
    proj=cv2.GaussianBlur(mag.mean(1),(51,1),0)
    y_peak=np.argmax(proj)
    mask=np.zeros_like(g);mask[y_peak,:]=255
    seed=(int(g.shape[1]*0.5),y_peak)
    mask=cv2.floodFill(cv2.cvtColor(g,cv2.COLOR_GRAY2BGR),
                       None,seed,255,
                       loDiff=(5,5,5),upDiff=(5,5,5))[1][:,:,0]
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c=max(cnts,key=cv2.contourArea)
    x,y,w,h=cv2.boundingRect(c)
    return {"x":x,"y":y,"w":w,"h":h,"poly":[{"x":x,"y":y},
                                            {"x":x+w,"y":y},
                                            {"x":x+w,"y":y+h},
                                            {"x":x,"y":y+h}]}

# ───── warp ─────
def warp(roi,poly):
    pts=np.array([[p["x"],p["y"]] for p in poly],dtype="float32")
    if len(pts)<4: return roi
    if len(pts)>4:pts=cv2.convexHull(pts)
    if len(pts)!=4:
        rect=cv2.boxPoints(cv2.minAreaRect(pts)).astype("float32")
    else: rect=order(pts)
    (tl,tr,br,bl)=rect
    W=int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl)))
    H=int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
    dst=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],dtype="float32")
    M=cv2.getPerspectiveTransform(rect,dst)
    return cv2.warpPerspective(roi,M,(W,H))

# ───── enhance ─────
def enhance(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab);l=cv2.createCLAHE(2.0,(16,16)).apply(l)
    img=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
    norm=cv2.divide(img,cv2.blur(img,(101,101)),scale=255)
    g=cv2.cvtColor(norm,cv2.COLOR_BGR2GRAY)
    bw=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                             cv2.THRESH_BINARY,35,5)
    if np.mean(bw[:30,:30])<128: bw=cv2.bitwise_not(bw)
    return bw

# ───── pipeline ─────
def process(img):
    if img.shape[0]>img.shape[1]:
        img=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)

    bbox=vision_bbox(img)
    if not bbox or grid_ratio(img[bbox["y"]:bbox["y"]+bbox["h"],
                                 bbox["x"]:bbox["x"]+bbox["w"]])<0.10:
        bbox=fft_flood(img)

    roi=img[bbox["y"]:bbox["y"]+bbox["h"],
            bbox["x"]:bbox["x"]+bbox["w"]]
    roi=warp(roi,bbox["poly"])
    out=enhance(roi)
    return out

# ───── routes ─────
@app.route("/process",methods=["POST"])
def proc():
    if "image" not in request.files:
        return jsonify(err="image?"),400
    data=request.files["image"].read()
    img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if img is None:return jsonify(err="decode"),400
    out=process(img);_,buf=cv2.imencode(".png",out)
    return send_file(BytesIO(buf.tobytes()),mimetype="image/png")

@app.route("/")
def home(): return "<h3>ECG‑API – grid straighten</h3>"

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",8080)),debug=False)