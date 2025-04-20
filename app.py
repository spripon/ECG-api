"""
ECG‑API – GPT‑4o Vision (+ HSV & FFT fallback)
Règles texte + symboles « U » inversés pour délimiter la zone quadrillée.
mai 2025 – spripon
"""

import os, base64, json, cv2, numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI

# ───────── CONFIG ─────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL          = os.getenv("MODEL_NAME", "gpt-4o")
MAX_DIM        = 1600
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
CORS(app)

# ────────── UTILS ──────────

def order_pts(pts):
    r=np.zeros((4,2),dtype="float32")
    s,d=pts.sum(1),np.diff(pts,1)
    r[0],r[2]=pts[np.argmin(s)],pts[np.argmax(s)]
    r[1],r[3]=pts[np.argmin(d)],pts[np.argmax(d)]
    return r

def warp(roi, poly):
    pts=np.array(poly,dtype="float32")
    if len(pts)!=4:
        pts=cv2.boxPoints(cv2.minAreaRect(pts)).astype("float32")
    tl,tr,br,bl=order_pts(pts)
    W=int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl)))
    H=int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
    dst=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],dtype="float32")
    M=cv2.getPerspectiveTransform(order_pts(pts),dst)
    return cv2.warpPerspective(roi,M,(W,H))

# ────── PROMPT VISION ──────
PROMPT=(
  "Repère la zone quadrillée ECG (rose/orange/rouge pâle).\n"
  "• Limite SUPÉRIEURE : ligne passant exactement entre – au-dessus –\n"
  "  le texte ‘12 dériv. ; position standard’ et – au-dessous – les labels D1/aVR\n"
  "  (ou ‘Unconfirmed diagnosis’ et V1/V4).\n"
  "• Limite INFÉRIEURE : ligne juste en dessous des textes ‘Vit. : 25 mm/s’,\n"
  "  ‘Pérph: 10 mm/mV’, ‘F 50- 0,15-100 Hz’.\n"
  "• Bord GAUCHE : se situe avant les étiquettes D1, D2, D3, D2 longue.\n"
  "• Bord DROIT : aligné sur les symboles en “U” inversé qui terminent V4‑V6\n"
  "  et la dérivation de rythme D2 longue.\n"
  "Renvoie STRICTEMENT un JSON {\"points\":[{\"x\":int,\"y\":int}]}\n"
  "avec les 4 coins (ordre horaire, haut‑gauche d’abord).\n"
  "Ignore bande blanche supérieure, marges, classeurs, ombres." )

# ────── DETECTION VISION ──────

def vision_bbox(img):
    if client is None: return None
    h,w=img.shape[:2];scale=1
    if max(h,w)>MAX_DIM:
        scale=MAX_DIM/max(h,w)
        img_r=cv2.resize(img,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)
    else: img_r=img
    _,buf=cv2.imencode(".png",img_r)
    b64=base64.b64encode(buf).decode()

    msgs=[{"role":"user","content":[
        {"type":"text","text":PROMPT},
        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]}]

    tools=[{"type":"function","function":{"name":"set",
        "parameters":{"type":"object","properties":{
          "points":{"type":"array","items":{
            "type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},
            "required":["x","y"]},"minItems":4,"maxItems":4}},
        "required":["points"]}}}]

    try:
        r=client.chat.completions.create(model=MODEL,messages=msgs,tools=tools,
            tool_choice={"type":"function","function":{"name":"set"}},temperature=0)
        pts=r.choices[0].message.tool_calls[0].function.arguments
        if isinstance(pts,str): pts=json.loads(pts)
        pts=pts["points"]
        xs,ys=[p["x"] for p in pts],[p["y"] for p in pts]
        x,y=min(xs),min(ys);w2,h2=max(xs)-x,max(ys)-y
        return {"x":int(x/scale),"y":int(y/scale),
                "w":int(w2/scale),"h":int(h2/scale),
                "poly":[{"x":int(p['x']/scale),"y":int(p['y']/scale)} for p in pts]}
    except Exception as e:
        print("Vision error:",e);return None

# ────── FALLBACK COULEUR ──────

def detect_color(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    low1,up1=np.array([0,25,40]),np.array([22,255,255])
    low2,up2=np.array([158,25,40]),np.array([179,255,255])
    m=cv2.inRange(hsv,low1,up1)|cv2.inRange(hsv,low2,up2)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    c=max(cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea)
    box=cv2.boxPoints(cv2.minAreaRect(c)).astype(int)
    xs,ys=box[:,0],box[:,1]
    return {"x":xs.min(),"y":ys.min(),"w":int(np.ptp(xs)),"h":int(np.ptp(ys)),
            "poly":[{"x":int(x),"y":int(y)} for x,y in box]}

# ────── ENHANCE NB ──────

def enhance(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,35,7)

# ────── PIPELINE ──────

def process(img):
    if img.shape[0]>img.shape[1]:
        img=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    box=vision_bbox(img) or detect_color(img)
    roi=img[box['y']:box['y']+box['h'], box['x']:box['x']+box['w']]
    roi=warp(roi,box['poly'])
    return enhance(roi)

# ────── FLASK ROUTES ──────
@app.route("/process",methods=["POST"])
def handle():
    if "image" not in request.files:
        return jsonify(error="image?"),400
    data=request.files["image"].read()
    img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    out=process(img);_,buf=cv2.imencode(".png",out)
    return send_file(BytesIO(buf.tobytes()),mimetype="image/png")

@app.route("/")
def home(): return "ECG‑API – zones délimitées par texte + symboles U"

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",8080)),debug=False)
