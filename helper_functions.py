# imports
import os
import cv2
import numpy as np
import json

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def preprocess_face_for_recognizer(gray_face, size=(90,120)):
    
    # deixa a imagem cinza
    if gray_face is None or gray_face.size == 0:
        raise ValueError("Empty face image")
    resized = cv2.resize(gray_face, size)
    equalized = cv2.equalizeHist(resized)
    return equalized

def estimate_brightness_from_frame(frame_bgr):
    
    # estimativa de iluminação dqa imagem
    try:
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]
        return float(np.mean(y))
    except Exception:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

# persistência em JSON
def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# carregamento de JSON
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
