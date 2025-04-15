from flask import Flask, request, send_file, jsonify # Ajout de jsonify pour les erreurs
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)
CORS(app) # Active CORS pour toutes les routes, ajustez si nécessaire pour plus de sécurité

# Fonction utilitaire pour ordonner les points du contour (coin sup-gauche, sup-droit, inf-droit, inf-gauche)
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)] # Point avec la plus petite somme (x+y) -> sup-gauche
    rect[2] = pts[np.argmax(s)] # Point avec la plus grande somme (x+y) -> inf-droit

    # Pour les points sup-droit et inf-gauche, on regarde la différence (y-x)
    rect[1] = pts[np.argmin(diff)] # Point avec la plus petite différence -> sup-droit
    rect[3] = pts[np.argmax(diff)] # Point avec la plus grande différence -> inf-gauche

    return rect

# Fonction principale de traitement de l'image ECG
def process_ecg(img_bgr):
    # 0. Copie de l'original pour le retour en cas d'échec partiel
    original_for_fallback = img_bgr.copy()

    # 1. Vérifier l'orientation et appliquer une rotation de -90 degrés si nécessaire
    # On suppose que l'ECG doit être plus large que haut (dérivations horizontales)
    h, w = img_bgr.shape[:2]
    rotated = False
    if h > w:
        print("Image détectée comme étant en portrait, rotation de -90 degrés.")
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated = True
        # Mettre à jour h et w après rotation
        h, w = img_bgr.shape[:2]

    # Garder une copie de l'image potentiellement tournée pour le warp final et la détection couleur
    original_rotated = img_bgr.copy()

    # --- DÉBUT DU BLOC INTÉGRÉ ---

    # 2. Détection couleur (pour trouver la zone d'intérêt) - PLAGES HSV OPTIMISÉES
    print("Détection de la couleur du quadrillage (plages HSV étendues)...")
    img_blurred = cv2.GaussianBlur(original_rotated, (5, 5), 0)
    hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)

    # --- Plages HSV ajustées pour plus de robustesse ---
    # Teinte (H): Les plages 0-15 (rouges/roses), 165-180 (rouges/roses) et 16-45(oranges/jaunes)
    #            sont conservées car elles couvrent bien les couleurs cibles.

    # Saturation (S): Abaissée à 20 (au lieu de 40) pour inclure les couleurs très pâles/délavées.
    #                 Maximum reste à 255 pour les couleurs vives/intenses.
    min_saturation = 20

    # Valeur (V): Abaissée à 90 (au lieu de 100) pour être un peu plus tolérant aux légères ombres
    #             ou aux impressions moins lumineuses. Maximum reste à 255.
    min_value = 90

    # Définition des nouvelles plages
    # [source: 14] Rouge/Rose (partie 1: près de 0 degrés)
    lower_red1 = np.array([0, min_saturation, min_value])
    upper_red1 = np.array([15, 255, 255])

    # [source: 15] Rouge/Rose (partie 2: près de 180 degrés)
    lower_red2 = np.array([165, min_saturation, min_value])
    upper_red2 = np.array([180, 255, 255])

    # [source: 16] Orange/Jaune
    lower_orange_yellow = np.array([16, min_saturation, min_value])
    # Vous pourriez étendre jusqu'à 50 ou 55 si certains jaunes tirent vers le vert
    upper_orange_yellow = np.array([45, 255, 255])

    # Créer les masques pour chaque plage
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_orange_yellow = cv2.inRange(hsv, lower_orange_yellow, upper_orange_yellow)

    # Combiner les masques
    print("Combinaison des masques de couleur...")
    combined_mask = cv2.bitwise_or(mask_red1, mask_red2)
    combined_mask = cv2.bitwise_or(combined_mask, mask_orange_yellow)

    # --- FIN DU BLOC INTÉGRÉ ---

    # 3. Nettoyage du masque et détection du contour du quadrillage
    print("Nettoyage du masque et recherche des contours...")
    # Utiliser des opérations morphologiques pour enlever le bruit et connecter les lignes du quadrillage
    kernel = np.ones((7, 7), np.uint8) # Noyau un peu plus grand pour mieux connecter
    mask_closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=2)

    # Trouver les contours externes dans le masque nettoyé
    # On se concentre sur le masque couleur, car Canny peut détecter les tracés ECG eux-mêmes
    contours, hierarchy = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Aucun contour trouvé basé sur la couleur du quadrillage.")
        # Fallback très simple : convertir l'image (potentiellement tournée) en N&B
        gray_fallback = cv2.cvtColor(original_rotated, cv2.COLOR_BGR2GRAY)
        return gray_fallback # Retourner l'image N&B sans recadrage/perspective

    # Trouver le contour le plus grand (on suppose que c'est le quadrillage)
    largest_contour = max(contours, key=cv2.contourArea)

    # Vérifier si le contour trouvé est raisonnablement grand
    min_area_ratio = 0.1 # Exiger que le contour occupe au moins 10% de l'image
    if cv2.contourArea(largest_contour) < min_area_ratio * h * w:
         print(f"Le plus grand contour trouvé est trop petit (aire: {cv2.contourArea(largest_contour)}). Utilisation de l'image complète.")
         # Fallback : convertir l'image (potentiellement tournée) en N&B
         gray_fallback = cv2.cvtColor(original_rotated, cv2.COLOR_BGR2GRAY)
         return gray_fallback

    # 4. Extraction de la zone du quadrillage et correction de perspective

    # Simplifier le contour pour obtenir les coins (si c'est un quadrilatère)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True) # 0.02 est un facteur d'epsilon courant

    warped = None
    # Si on a bien 4 coins, on applique la correction de perspective
    if len(approx) == 4:
        print("Contour à 4 points détecté, application de la correction de perspective.")
        pts = np.array([p[0] for p in approx], dtype="float32")
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # Calculer la largeur et la hauteur de la nouvelle image redressée
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        # Définir les points de destination pour l'image redressée (vue de dessus)
        dst = np.array([
            [0, 0],                  # Coin sup-gauche
            [maxWidth - 1, 0],       # Coin sup-droit
            [maxWidth - 1, maxHeight - 1], # Coin inf-droit
            [0, maxHeight - 1]       # Coin inf-gauche
            ], dtype="float32")

        # Calculer la matrice de transformation perspective
        M = cv2.getPerspectiveTransform(rect, dst)
        # Appliquer la transformation à l'image originale (potentiellement tournée)
        warped = cv2.warpPerspective(original_rotated, M, (maxWidth, maxHeight))

    else:
        print(f"Contour détecté avec {len(approx)} points (pas 4). Utilisation du rectangle englobant.")
        # Fallback si ce n'est pas un quadrilatère : utiliser le rectangle englobant
        x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
        # Recadrer l'image originale (potentiellement tournée)
        # Ajouter un petit padding pour ne pas couper les bords si le boundingRect est trop juste
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + w_rect + padding) # w est la largeur de original_rotated
        y2 = min(h, y + h_rect + padding) # h est la hauteur de original_rotated
        warped = original_rotated[y1:y2, x1:x2]

    # 5. Conversion finale en monochrome (Noir et Blanc)
    if warped is not None and warped.size > 0 :
        print("Conversion de l'image traitée en niveaux de gris.")
        final_image = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # Optionnel: Améliorer le contraste si nécessaire (peut aider pour la lecture des tracés)
        # final_image = cv2.equalizeHist(final_image) # Histogram Equalization
        # Ou utiliser CLAHE pour une amélioration locale du contraste
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # final_image = clahe.apply(final_image)
    else:
        print("Warping a échoué ou produit une image vide. Retour à l'image grise de secours.")
        final_image = cv2.cvtColor(original_rotated, cv2.COLOR_BGR2GRAY) # Fallback N&B de l'image tournée

    return final_image

@app.route("/", methods=["GET"])
def index():
    # Simple page d'accueil pour vérifier que l'API tourne
    return "<h3>API de traitement ECG est active. Utilisez POST /process avec une image.</h3>"

@app.route("/process", methods=["POST"])
def process_image_route():
    # Vérifier si un fichier 'image' est présent dans la requête
    if 'image' not in request.files:
        print("Erreur: Aucune image fournie dans la requête.")
        return jsonify({"error": "Aucun fichier image fourni ('image' attendu)"}), 400

    file = request.files['image']

    # Vérifier si le fichier a un nom (basique)
    if file.filename == '':
        print("Erreur: Nom de fichier vide.")
        return jsonify({"error": "Nom de fichier vide"}), 400

    try:
        # Lire les données de l'image depuis le buffer en mémoire
        img_data = file.read()
        # Convertir les données brutes en un tableau numpy
        np_arr = np.frombuffer(img_data, np.uint8)
        # Décoder le tableau numpy en une image OpenCV (en couleur BGR par défaut)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            print("Erreur: Impossible de décoder l'image. Format non supporté ou fichier corrompu?")
            return jsonify({"error": "Impossible de décoder le fichier image"}), 400

        # Appeler la fonction de traitement principale
        processed_image_gray = process_ecg(img_bgr)

        # Vérifier si le traitement a retourné une image valide
        if processed_image_gray is None or processed_image_gray.size == 0:
             print("Erreur: Le traitement n'a pas produit d'image valide.")
             return jsonify({"error": "Échec du traitement de l'image"}), 500

        # Encoder l'image traitée (monochrome) en format JPEG pour l'envoi
        # Utiliser PNG pourrait être mieux pour du N&B sans perte, mais JPEG est courant
        is_success, buffer = cv2.imencode(".jpg", processed_image_gray, [cv2.IMWRITE_JPEG_QUALITY, 90]) # Qualité 90

        if not is_success:
            print("Erreur: Impossible d'encoder l'image traitée en JPEG.")
            return jsonify({"error": "Échec de l'encodage de l'image résultat"}), 500

        # Créer un objet BytesIO pour envoyer le buffer comme un fichier
        img_byte_io = BytesIO(buffer.tobytes())

        # Envoyer le fichier image en réponse
        print("Envoi de l'image traitée.")
        return send_file(
            img_byte_io,
            mimetype='image/jpeg',
            as_attachment=False # Envoyer inline plutôt qu'en téléchargement
            # download_name='processed_ecg.jpg' # Nom si as_attachment=True
        )

    except Exception as e:
        # Capturer les erreurs potentielles pendant la lecture ou le traitement
        print(f"Erreur serveur lors du traitement de l'image: {e}")
        import traceback
        traceback.print_exc() # Affiche la trace complète dans les logs serveur
        return jsonify({"error": f"Erreur interne du serveur: {e}"}), 500

if __name__ == "__main__":
    # Récupérer le port depuis les variables d'environnement (pour Railway/Heroku etc.)
    # Utiliser 8080 par défaut si PORT n'est pas défini
    port = int(os.environ.get("PORT", 8080))
    # Lancer l'application Flask
    # host='0.0.0.0' permet d'écouter sur toutes les interfaces réseau disponibles
    print(f"Démarrage du serveur Flask sur le port {port}")
    app.run(host="0.0.0.0", port=port, debug=False) # Mettre debug=True SEULEMENT en développement local

