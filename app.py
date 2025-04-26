import os, base64, json, cv2, numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VISION_MODEL   = os.getenv("MODEL_NAME", "gpt-4o")
MAX_DIM    = 1600
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
CORS(app)

# ---- geometry ----
def order(pts):
    rect = np.zeros((4,2),dtype="float32")
    s,d  = pts.sum(1), np.diff(pts,1)
    rect[0],rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1],rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

def warp(img, poly):
    """
    Transforme une image trapézoïdale en rectangle parfait avec des angles à 90 degrés
    en utilisant les 4 coins détectés.
    """
    # Convertir les points en tableau numpy
    pts = np.array([[p["x"], p["y"]] for p in poly], dtype="float32")

    # Si moins de 4 points, utiliser le rectangle minimum englobant
    if pts.shape[0] != 4:
        pts = cv2.boxPoints(cv2.minAreaRect(pts)).astype("float32")

    # Ordonner les points (haut-gauche, haut-droite, bas-droite, bas-gauche)
    rect = order(pts)

    # Calculer les dimensions du rectangle de sortie
    # Utiliser la largeur et hauteur maximales pour garantir un rectangle parfait
    widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
    heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Maintenir un rapport d'aspect cohérent pour les ECG
    # Les ECG standard ont généralement un rapport largeur/hauteur d'environ 3:1 à 4:1
    aspect_ratio = maxWidth / maxHeight
    if aspect_ratio < 2.5:  # Si trop étroit
        maxWidth = int(maxHeight * 3)  # Forcer un rapport de 3:1
    elif aspect_ratio > 4.5:  # Si trop large
        maxHeight = int(maxWidth / 3)  # Forcer un rapport de 3:1

    # Construire le rectangle de destination avec des angles parfaitement droits
    dst = np.array([
        [0, 0],                      # haut-gauche
        [maxWidth - 1, 0],           # haut-droite
        [maxWidth - 1, maxHeight - 1], # bas-droite
        [0, maxHeight - 1]           # bas-gauche
    ], dtype="float32")

    # Calculer la matrice de transformation homographique
    M = cv2.getPerspectiveTransform(rect, dst)

    # Appliquer la transformation de perspective avec interpolation cubique
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

    return warped

def detect_grid_corners(img):
    """
    Fonction combinée pour détecter précisément les 4 coins de la zone quadrillée ECG
    en utilisant à la fois la détection de couleur et l'analyse d'image avancée.
    """
    # 1. Détection initiale par couleur pour isoler la zone quadrillée
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Plages de couleurs élargies pour couvrir toutes les nuances possibles du papier ECG
    # Rouge/Rose
    lower_red1 = np.array([0, 20, 180])
    upper_red1 = np.array([20, 150, 255])
    lower_red2 = np.array([160, 20, 180])
    upper_red2 = np.array([179, 150, 255])
    
    # Jaune/Orange
    lower_yellow = np.array([20, 20, 180])
    upper_yellow = np.array([40, 150, 255])
    
    # Orange
    lower_orange = np.array([10, 50, 180])
    upper_orange = np.array([25, 150, 255])

    # Créer des masques pour chaque plage de couleur
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combiner tous les masques
    mask_combined = cv2.bitwise_or(mask_red1, mask_red2)
    mask_combined = cv2.bitwise_or(mask_combined, mask_yellow)
    mask_combined = cv2.bitwise_or(mask_combined, mask_orange)

    # Appliquer des opérations morphologiques pour nettoyer le masque
    kernel = np.ones((5, 5), np.uint8)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)

    # 2. Détection des contours de la zone quadrillée
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Trouver le plus grand contour (zone quadrillée ECG)
    largest_contour = max(contours, key=cv2.contourArea)

    # 3. Analyse plus fine des bords pour trouver les coins précis
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Appliquer un filtre de détection de bord
    edges = cv2.Canny(gray, 50, 150)

    # Créer un masque pour ne considérer que les bords dans la région d'intérêt
    mask_roi = np.zeros_like(edges)
    cv2.drawContours(mask_roi, [largest_contour], 0, 255, -1)
    edges_roi = cv2.bitwise_and(edges, mask_roi)

    # 4. Détection des coins par analyse des gradients
    # Appliquer l'algorithme de Harris Corner Detection
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)

    # Créer un masque pour ne considérer que les coins dans la région d'intérêt
    corners_roi = cv2.bitwise_and(corners > 0.01 * corners.max(), mask_roi)

    # Convertir les positions des coins en coordonnées
    corner_points = np.argwhere(corners_roi > 0)

    if len(corner_points) < 4:
        # Si pas assez de coins détectés, utiliser l'approximation polygonale
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Si l'approximation donne trop de points, prendre les 4 coins du rectangle englobant
        if len(approx) != 4:
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            corner_points = box
    else:
        # Regrouper les points de coin proches (clustering)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(corner_points.astype(np.float32), 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        corner_points = centers

    # 5. Vérification et ajustement des coins
    # Si nous avons plus ou moins de 4 points, utiliser l'approximation du rectangle
    if len(corner_points) != 4:
        rect = cv2.minAreaRect(largest_contour)
        corner_points = cv2.boxPoints(rect)

    # Convertir en format attendu par la fonction warp
    corners_list = [{"x": int(p[1]) if len(p) > 1 else int(p[0]),
                     "y": int(p[0]) if len(p) > 1 else int(p[1])}
                    for p in corner_points]

    # 6. Utiliser l'API Vision si disponible pour validation/amélioration
    if client is not None:
        vision_corners = detect_vision_corners(img)
        if vision_corners:
            # Combiner les résultats (moyenne pondérée)
            for i in range(min(len(corners_list), len(vision_corners))):
                corners_list[i]["x"] = int((corners_list[i]["x"] * 0.7 + vision_corners[i]["x"] * 0.3))
                corners_list[i]["y"] = int((corners_list[i]["y"] * 0.7 + vision_corners[i]["y"] * 0.3))

    # Calculer les dimensions approximatives
    xs = [p["x"] for p in corners_list]
    ys = [p["y"] for p in corners_list]

    return {
        "poly": corners_list,
        "x": min(xs),
        "y": min(ys),
        "w": max(xs) - min(xs),
        "h": max(ys) - min(ys)
    }

def detect_vision_corners(img):
    """
    Utilise l'API Vision pour détecter les coins de la zone quadrillée
    """
    if client is None:
        return None

    h, w = img.shape[:2]
    scale = 1
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf).decode()

    msg = [{"role": "user", "content": [
        {"type": "text", "text": "Identifie précisément les 4 coins (supérieur gauche, supérieur droit, inférieur droit, inférieur gauche) de la zone quadrillée de l'ECG. La zone peut être de couleur rose, rouge pâle, jaune pâle, orange ou toute nuance intermédiaire. Donne les coordonnées x,y de chaque coin dans cet ordre."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    ]}]

    tools = [{"type": "function", "function": {"name": "set_corners", "parameters": {
        "type": "object", "properties": {
            "corners": {"type": "array", "items": {
                "type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"]
            }, "minItems": 4, "maxItems": 4}
        },
        "required": ["corners"]
    }}}]

    r = client.chat.completions.create(
        model=VISION_MODEL,
        messages=msg,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "set_corners"}},
        temperature=0
    )

    args = r.choices[0].message.tool_calls[0].function.arguments
    if isinstance(args, str):
        args = json.loads(args)

    corners = args["corners"]
    return [{"x": int(c["x"] / scale), "y": int(c["y"] / scale)} for c in corners]

# ---- colour detection (méthode originale comme fallback) ----
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

# ---- Vision fallback (méthode originale comme fallback) ----
def detect_vision(img):
    if client is None: return None
    h,w = img.shape[:2]; scale=1
    if max(h,w)>MAX_DIM:
        scale=MAX_DIM/max(h,w)
        img=cv2.resize(img,(int(w*scale),int(h*scale)))
    _,buf=cv2.imencode(".png",img)
    b64=base64.b64encode(buf).decode()
    msg=[{"role":"user","content":[
    {"type":"text","text":"Donne 4‑8 points de la zone quadrillée ECG."},
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

# ---- iPhone‑like enhancement ----
def enhance(img):
    # exposition + brillance
    exp=cv2.convertScaleAbs(img,alpha=1.2,beta=20)
    # désaturation
    hsv=cv2.cvtColor(exp,cv2.COLOR_BGR2HSV)
    hsv[:,:,1]=(hsv[:,:,1]*0.5).astype(np.uint8)
    exp=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # suppression des ombres
    bg=cv2.medianBlur(exp,51)
    flat=cv2.divide(exp,bg,scale=255)
    # sharpen
    sh=cv2.addWeighted(flat,1.3,cv2.GaussianBlur(flat,(0,0),3),-0.3,0)
    return sh

# ---- pipeline ----
def process(img):
    """
    Pipeline complet de traitement de l'image ECG
    """
    # Rotation si nécessaire
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Utiliser notre nouvelle fonction de détection des coins
    box = detect_grid_corners(img)

    # Si la détection échoue, essayer les méthodes précédentes comme fallback
    if not box:
        box = detect_color(img) or detect_vision(img)

    if not box:
        raise ValueError("zone non trouvée")

    # Appliquer la transformation de perspective
    roi = warp(img, box["poly"])

    # Améliorer l'image
    return enhance(roi)

# ---- Flask ----
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
def home(): return "<h3>ECG‑API filtre argenté</h3>"

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",8080)),debug=False)
