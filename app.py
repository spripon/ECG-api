# -*- coding: utf-8 -*-
# Version optimisée le 2025-04-15

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
import os
import traceback # Pour un meilleur logging des erreurs

app = Flask(__name__)
# Configuration CORS (ajuster 'origins' pour plus de sécurité en production si nécessaire)
CORS(app, resources={r"/process": {"origins": "*"}}) # Permet les requêtes POST sur /process depuis n'importe quelle origine

# Fonction utilitaire pour ordonner les points du contour (TL, TR, BR, BL)
def order_points(pts):
    """Ordonne les 4 points d'un contour pour la transformation de perspective."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)] # Coin Supérieur Gauche (Top-Left)
    rect[2] = pts[np.argmax(s)] # Coin Inférieur Droit (Bottom-Right)
    rect[1] = pts[np.argmin(diff)] # Coin Supérieur Droit (Top-Right)
    rect[3] = pts[np.argmax(diff)] # Coin Inférieur Gauche (Bottom-Left)
    return rect

# Fonction pour l'amélioration du contraste et la binarisation
def enhance_and_binarize(image_gray):
    """Applique CLAHE et seuillage adaptatif pour améliorer le contraste et binariser."""
    if image_gray is None or image_gray.size == 0:
        print("DEBUG: enhance_and_binarize reçu une image vide.")
        return None

    h, w = image_gray.shape
    print(f"DEBUG: enhance_and_binarize - Dimensions image entrée: {w}x{h}")

    # 1. Amélioration du contraste local avec CLAHE
    print("DEBUG: Application de CLAHE...")
    try:
        # clipLimit: Limite l'amplification du contraste (évite le bruit excessif)
        # tileGridSize: Taille de la grille locale pour l'égalisation
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)) # clipLimit légèrement augmenté
        enhanced_gray = clahe.apply(image_gray)
        print("DEBUG: CLAHE appliqué.")
    except Exception as e:
        print(f"ERREUR: Échec de l'application CLAHE: {e}")
        return image_gray # Retourner l'image grise originale si CLAHE échoue

    # 2. Binarisation avec seuillage adaptatif
    print("DEBUG: Application du seuillage adaptatif...")
    try:
        # blockSize: Taille du voisinage (impair). Ajuster si les lignes sont trop fines/épaisses.
        # C: Constante soustraite de la moyenne. Ajuster pour plus/moins de noir.
        block_size = 15 # Taille de voisinage
        C_value = 4     # Constante de soustraction (légèrement augmentée)
        binary_image = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, block_size, C_value) # THRESH_BINARY_INV pour fond blanc/tracé noir
        print("DEBUG: Seuil adaptatif appliqué.")

        # Optionnel : Nettoyage mineur du bruit (peut enlever des petits points)
        # kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_clean, iterations=1)
        # print("DEBUG: Nettoyage morphologique (open) appliqué.")

    except Exception as e:
        print(f"ERREUR: Échec du seuillage adaptatif: {e}")
        # Fallback: essayer un seuil global Otsu sur l'image améliorée par CLAHE
        try:
             print("DEBUG: Tentative de fallback avec seuil Otsu...")
             _, binary_image = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
             print("DEBUG: Seuil Otsu appliqué avec succès.")
        except Exception as e_otsu:
             print(f"ERREUR: Échec du seuil Otsu également: {e_otsu}")
             return enhanced_gray # Dernier recours: retourner l'image CLAHE

    return binary_image

# Fonction principale de traitement de l'image ECG
def process_ecg(img_bgr):
    """Traite une image ECG pour détecter, redresser, et améliorer le quadrillage."""
    print("--- Début du traitement ECG ---")
    # 0. Copie et Vérification initiale
    if img_bgr is None or img_bgr.size == 0:
        print("ERREUR: process_ecg - Image d'entrée invalide.")
        return None
    # Garder une copie pour le fallback ultime si tout échoue
    original_for_ultimate_fallback = img_bgr.copy()

    # 1. Rotation si nécessaire (pour orientation paysage)
    h_orig, w_orig = img_bgr.shape[:2]
    print(f"DEBUG: Dimensions originales: {w_orig}x{h_orig}")
    if h_orig > w_orig * 1.1: # Rotation si significativement plus haut que large
        print("INFO: Rotation de l'image de -90 degrés.")
        try:
            img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception as e:
            print(f"ERREUR: Echec de la rotation: {e}")
            img_bgr = original_for_ultimate_fallback # Revenir à l'original si rotation échoue
    original_rotated = img_bgr.copy()
    h, w = original_rotated.shape[:2]
    print(f"DEBUG: Dimensions après rotation potentielle: {w}x{h}")

    # --- Fonction Fallback interne ---
    def get_enhanced_fallback(image_to_process):
        print("INFO: Utilisation du fallback: Amélioration de l'image entière (sans recadrage/perspective).")
        try:
            gray_fallback = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
            enhanced_fallback = enhance_and_binarize(gray_fallback)
            return enhanced_fallback
        except Exception as e_fallback:
             print(f"ERREUR: Échec critique dans get_enhanced_fallback: {e_fallback}")
             # Ultime recours: convertir l'original en gris simple
             try:
                 return cv2.cvtColor(original_for_ultimate_fallback, cv2.COLOR_BGR2GRAY)
             except:
                 return None # Si même ça échoue...
    # --- Fin Fonction Fallback ---

    # 2. Détection couleur (avec plages HSV optimisées)
    print("INFO: Détection de la couleur du quadrillage (plages HSV étendues)...")
    try:
        img_blurred = cv2.GaussianBlur(original_rotated, (5, 5), 0)
        hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)

        # Paramètres HSV optimisés
        min_saturation = 15 # Encore plus bas pour les couleurs très très pâles
        min_value = 85      # Encore plus bas pour tolérer plus d'ombre

        # Rouge/Rose (partie 1)
        lower_red1 = np.array([0, min_saturation, min_value])
        upper_red1 = np.array([15, 255, 255])
        # Rouge/Rose (partie 2)
        lower_red2 = np.array([165, min_saturation, min_value])
        upper_red2 = np.array([180, 255, 255])
        # Orange/Jaune
        lower_orange_yellow = np.array([16, min_saturation, min_value])
        upper_orange_yellow = np.array([50, 255, 255]) # Étendu jusqu'à H=50

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_orange_yellow = cv2.inRange(hsv, lower_orange_yellow, upper_orange_yellow)

        combined_mask = cv2.bitwise_or(mask_red1, mask_red2)
        combined_mask = cv2.bitwise_or(combined_mask, mask_orange_yellow)
        print("DEBUG: Masque de couleur combiné créé.")

    except Exception as e_color:
        print(f"ERREUR: Échec lors de la détection couleur: {e_color}")
        return get_enhanced_fallback(original_rotated) # Fallback si détection couleur échoue

    # 3. Nettoyage masque et détection contour
    print("INFO: Nettoyage du masque et recherche des contours...")
    try:
        # Noyau plus adapté pour connecter lignes fines mais pas fusionner de trop grandes zones
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)) # Rectangulaire pour connecter horizontalement
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_processed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_OPEN, kernel_open, iterations=2)

        contours, hierarchy = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"DEBUG: {len(contours)} contours trouvés initialement.")

        if not contours:
            print("AVERTISSEMENT: Aucun contour trouvé basé sur la couleur.")
            # Tentative avec Canny comme secours avant fallback complet? Non, simplifions.
            return get_enhanced_fallback(original_rotated)

        # Filtrer les contours trop petits avant de chercher le plus grand
        min_contour_area = 0.05 * h * w # Exiger au moins 5% de l'aire de l'image
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        if not large_contours:
            print("AVERTISSEMENT: Aucun contour suffisamment grand trouvé.")
            return get_enhanced_fallback(original_rotated)

        largest_contour = max(large_contours, key=cv2.contourArea)
        print(f"DEBUG: Aire du plus grand contour: {cv2.contourArea(largest_contour)}")

    except Exception as e_contour:
        print(f"ERREUR: Échec lors du traitement des contours: {e_contour}")
        return get_enhanced_fallback(original_rotated)

    # 4. Correction de Perspective / Recadrage
    print("INFO: Tentative de correction de perspective...")
    warped = None
    try:
        peri = cv2.arcLength(largest_contour, True)
        # Utiliser un epsilon relatif plus petit peut aider à préserver les 4 coins
        approx = cv2.approxPolyDP(largest_contour, 0.015 * peri, True) # Epsilon ajusté

        if len(approx) == 4:
            print(f"INFO: Contour à 4 points trouvé ({len(approx)} points). Application du warp perspective.")
            pts = np.array([p[0] for p in approx], dtype="float32")
            rect = order_points(pts)
            (tl, tr, br, bl) = rect

            # Calcul dimensions de sortie (méthode stable)
            widthA = np.linalg.norm(br - bl); widthB = np.linalg.norm(tr - tl)
            maxWidth = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br); heightB = np.linalg.norm(tl - bl)
            maxHeight = int(max(heightA, heightB))

            if maxWidth > 0 and maxHeight > 0:
                dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(original_rotated, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
                print("DEBUG: Warp perspective appliqué.")
            else:
                 print("AVERTISSEMENT: Dimensions calculées pour warp sont invalides.")
                 warped = None # Forcer le fallback au bounding box

        # Si approx n'a pas 4 points OU si warp a échoué
        if warped is None:
             if len(approx) != 4:
                 print(f"INFO: Contour trouvé avec {len(approx)} points. Utilisation du rectangle englobant.")
             else: # warped est None mais len(approx) == 4, erreur de calcul dims?
                 print("AVERTISSEMENT: Echec du warp malgré 4 points détectés. Utilisation du rectangle englobant.")

             x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
             # Recadrage direct sans padding pour maximiser la zone détectée
             if h_rect > 0 and w_rect > 0:
                 warped = original_rotated[y:y + h_rect, x:x + w_rect]
                 print("DEBUG: Recadrage par Bounding Box appliqué.")
             else:
                 print("AVERTISSEMENT: Rectangle englobant invalide.")
                 warped = None # Échec du recadrage aussi

    except Exception as e_warp:
        print(f"ERREUR: Échec lors du redressement/recadrage: {e_warp}")
        warped = None # Assurer que warped est None pour déclencher le fallback final

    # 5. Conversion N&B et Amélioration finale
    if warped is not None and warped.size > 0:
        print("INFO: Conversion en N&B et Amélioration de l'image redressée/recadrée.")
        try:
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            final_image = enhance_and_binarize(warped_gray)
        except Exception as e_enhance:
            print(f"ERREUR: Échec de l'amélioration finale sur l'image redressée: {e_enhance}")
            final_image = None # Déclenchera le fallback ci-dessous

        if final_image is None or final_image.size == 0:
             print("AVERTISSEMENT: L'amélioration a échoué. Utilisation du fallback.")
             final_image = get_enhanced_fallback(original_rotated) # Appliquer fallback sur l'image tournée
        else:
             print("INFO: Traitement terminé avec succès (image redressée/améliorée).")

    else:
        # Si warped est None ou vide (échec perspective ET bounding box)
        print("AVERTISSEMENT: Échec du redressement ET du recadrage. Utilisation du fallback sur image entière.")
        final_image = get_enhanced_fallback(original_rotated)

    print("--- Fin du traitement ECG ---")
    return final_image

# --- Routes Flask ---
@app.route("/", methods=["GET"])
def index():
    """Page d'accueil simple pour vérifier que l'API tourne."""
    return "<h3>API de traitement ECG v2.1 (HSV optimisé) est active. Utilisez POST /process avec une image.</h3>", 200

@app.route("/process", methods=["POST"])
def process_image_route():
    """Route principale pour recevoir une image et retourner l'ECG traité."""
    print("--- Requête reçue sur /process ---")
    if 'image' not in request.files:
        print("ERREUR API: Aucun fichier 'image' fourni.")
        return jsonify({"error": "Aucun fichier image fourni ('image' attendu)"}), 400

    file = request.files['image']

    if not file or file.filename == '':
        print("ERREUR API: Fichier image invalide ou nom de fichier vide.")
        return jsonify({"error": "Fichier image invalide ou nom de fichier vide"}), 400

    print(f"INFO: Fichier reçu: {file.filename}, Type: {file.mimetype}")

    try:
        # Lire en mémoire
        img_data = file.read()
        np_arr = np.frombuffer(img_data, np.uint8)
        # Décoder l'image en couleur (BGR)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            print("ERREUR API: Impossible de décoder l'image. Format non supporté ou fichier corrompu?")
            return jsonify({"error": "Impossible de décoder le fichier image"}), 400

        # --- Appel de la fonction de traitement principale ---
        processed_image_binary = process_ecg(img_bgr)
        # ----------------------------------------------------

        if processed_image_binary is None or processed_image_binary.size == 0:
             print("ERREUR API: Le traitement interne n'a pas produit d'image valide.")
             return jsonify({"error": "Échec du traitement interne de l'image"}), 500

        # Encoder l'image finale (binaire N&B) en PNG (sans perte)
        is_success, buffer = cv2.imencode(".png", processed_image_binary)

        if not is_success:
            print("ERREUR API: Impossible d'encoder l'image traitée en PNG.")
            return jsonify({"error": "Échec de l'encodage de l'image résultat"}), 500

        # Créer un objet BytesIO pour envoyer le buffer comme un fichier
        img_byte_io = BytesIO(buffer.tobytes())

        print("INFO: Envoi de l'image PNG traitée.")
        return send_file(
            img_byte_io,
            mimetype='image/png',
            as_attachment=False # Envoyer 'inline'
            # download_name='processed_ecg.png' # Optionnel si as_attachment=True
        )

    except cv2.error as cv_err:
         print(f"ERREUR API: Erreur OpenCV: {cv_err}")
         traceback.print_exc()
         return jsonify({"error": f"Erreur lors du traitement d'image OpenCV: {cv_err}"}), 500
    except Exception as e:
        # Capturer les autres erreurs potentielles
        print(f"ERREUR API: Erreur serveur inattendue: {e}")
        traceback.print_exc() # Affiche la trace complète dans les logs serveur
        return jsonify({"error": f"Erreur interne du serveur: {e}"}), 500

# Point d'entrée pour l'exécution directe ou via un serveur WSGI (comme Gunicorn)
if __name__ == "__main__":
    # Récupérer le port depuis les variables d'environnement (pour Railway, Heroku, etc.)
    port = int(os.environ.get("PORT", 8080))
    # Lancer l'application Flask en mode développement (debug=False pour prod)
    # host='0.0.0.0' est important pour écouter sur toutes les interfaces dans un conteneur
    print(f"INFO: Démarrage du serveur Flask sur http://0.0.0.0:{port}")
    # Mettre debug=True SEULEMENT pour le développement local, JAMAIS en production.
    app.run(host="0.0.0.0", port=port, debug=False)
