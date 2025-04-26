import os, base64, json, cv2, numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VISION_MODEL   = os.getenv("MODEL_NAME", "gpt-4o")
MAX_DIM        = 1600
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
CORS(app)

# ---------- geometry ----------
def order(pts):
    rect = np.zeros((4,2),dtype="float32")
    s,d  = pts.sum(1), np.diff(pts,1)
    rect[0],rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1],rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

def warp(img, poly):
    pts = np.array([[p["x"],p["y"]] for p in poly],dtype="float32")
    if pts.shape[0]!=4:
        pts = cv2.boxPoints(cv2.minAreaRect(pts)).astype("float32")
    tl,tr,br,bl = order(pts)
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],dtype="float32")
    M   = cv2.getPerspectiveTransform(order(pts),dst)
    return cv2.warpPerspective(img,M,(W,H))

# ---------- colour detection ----------
def detect_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1,upper1 = np.array([  0,30,50]), np.array([ 20,255,255])
    lower2,upper2 = np.array([160,30,50]), np.array([179,255,255])
    mask = cv2.inRange(hsv,lower1,upper1)|cv2.inRange(hsv,lower2,upper2)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((25,25),np.uint8))
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    pts = cv2.boxPoints(cv2.minAreaRect(max(cnts,key=cv2.contourArea))).astype(int)
    xs,ys = pts[:,0], pts[:,1]
    return {"poly":[{"x":int(x),"y":int(y)} for x,y in pts],
            "x":int(xs.min()),"y":int(ys.min()),
            "w":int(np.ptp(xs)),"h":int(np.ptp(ys))}

# ---------- Vision fallback ----------
def detect_vision(img):
    if client is None: return None
    h,w = img.shape[:2]; scale=1
    if max(h,w)>MAX_DIM:
        scale=MAX_DIM/max(h,w)
        img=cv2.resize(img,(int(w*scale),int(h*scale)))
    _,buf=cv2.imencode(".png",img)
    b64=base64.b64encode(buf).decode()
    msg=[{"role":"user","content":[
        {"type":"text","text":"Donne 4â8 points de la zone quadrillÃ©e ECG."},
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

# ---------- iPhoneâlike enhancement ----------
def enhance(img):
    # exposition + brillance
    exp=cv2.convertScaleAbs(img,alpha=1.2,beta=20)
    # dÃ©saturation
    hsv=cv2.cvtColor(exp,cv2.COLOR_BGR2HSV)
    hsv[:,:,1]=(hsv[:,:,1]*0.5).astype(np.uint8)
    exp=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # suppression des ombres
    bg=cv2.medianBlur(exp,51)
    flat=cv2.divide(exp,bg,scale=255)
    # sharpen
    sh=cv2.addWeighted(flat,1.3,cv2.GaussianBlur(flat,(0,0),3),-0.3,0)
    return sh

# ---------- pipeline ----------
def process(img):
    if img.shape[0]>img.shape[1]:
        img=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    box=detect_color(img) or detect_vision(img)
    if not box: raise ValueError("zone non trouvÃ©e")
    roi = warp(img,box["poly"])
    return enhance(roi)

# ---------- Flask ----------
app=Flask(__name__); CORS(app)

@app.route("/process",methods=["POST"])
def handle():
    if "image" not in request.files:
        return jsonify(error="image?"),400
    img=cv2.imdecode(np.frombuffer(request.files["image"].read(),np.uint8),
                     cv2.IMREAD_COLOR)
    try:
        out=process(img);_,buf=cv2.imencode(".png",out)
        return send_file(BytesIO(buf.tobytes()),mimetype="image/png")
    except Exception as e:
        return jsonify(error=str(e)),500

@app.route("/")
def home(): return "<h3>ECGâAPI filtre argentÃ©</h3>"

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",8080)),debug=False)