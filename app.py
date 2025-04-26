import os, base64, json, cv2, numpy as np
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI

# ... (garder les configurations initiales identiques)

def preprocess_image(img):
    # Améliorer le contraste
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Réduire le bruit
    img = cv2.GaussianBlur(img, (3,3), 0)
    return img

def order(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def auto_canny(image, sigma=0.33):
    # Détection automatique des contours
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

def detect_color(img):
    # Conversion en HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Plage de couleurs rose élargie
    lower1 = np.array([150, 20, 50])
    upper1 = np.array([179, 255, 255])
    lower2 = np.array([0, 20, 50])
    upper2 = np.array([20, 255, 255])
    
    # Création du masque
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Amélioration du masque
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Détection des contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    
    # Sélection du plus grand contour
    cnt = max(cnts, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Si on n'a pas exactement 4 points, utiliser minAreaRect
    if len(approx) != 4:
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect)
    else:
        pts = approx.reshape(-1, 2)
    
    pts = pts.astype(int)
    xs, ys = pts[:,0], pts[:,1]
    return {
        "poly": [{"x": int(x), "y": int(y)} for x, y in pts],
        "x": int(xs.min()),
        "y": int(ys.min()),
        "w": int(np.ptp(xs)),
        "h": int(np.ptp(ys))
    }

def warp(img, poly):
    # Obtention des points
    pts = np.array([[p["x"], p["y"]] for p in poly], dtype="float32")
    if pts.shape[0] != 4:
        pts = cv2.boxPoints(cv2.minAreaRect(pts)).astype("float32")
    
    # Ordonner les points
    rect = order(pts)
    (tl, tr, br, bl) = rect
    
    # Calculer les dimensions maximales
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Créer les points de destination
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Calculer la matrice de transformation
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    # Correction de l'angle si nécessaire
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[0]:
            angle = theta * 180 / np.pi
            if angle < 45:
                angles.append(angle)
            elif angle > 135:
                angles.append(angle - 180)
        
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                center = (warped.shape[1] // 2, warped.shape[0] // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                warped = cv2.warpAffine(warped, M, (warped.shape[1], warped.shape[0]))
    
    return warped

def enhance(img):
    # Amélioration du contraste
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Suppression des ombres
    bg = cv2.medianBlur(enhanced, 51)
    flat = cv2.divide(enhanced, bg, scale=255)
    
    # Amélioration de la netteté
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharp = cv2.filter2D(flat, -1, kernel)
    
    # Ajustement final
    final = cv2.convertScaleAbs(sharp, alpha=1.1, beta=10)
    return final

def process(img):
    # Rotation si nécessaire
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Prétraitement
    img = preprocess_image(img)
    
    # Détection et correction
    box = detect_color(img) or detect_vision(img)
    if not box:
        raise ValueError("zone non trouvée")
    
    # Correction de perspective et amélioration
    roi = warp(img, box["poly"])
    return enhance(roi)

# ... (garder le reste du code Flask identique)
