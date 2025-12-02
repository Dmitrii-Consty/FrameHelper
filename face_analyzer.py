# imports
import os
import cv2
import numpy as np
import mediapipe as mp
from helper_functions import preprocess_face_for_recognizer, estimate_brightness_from_frame

# módulos locais
SSD_PROTO = "deploy.prototxt.txt"
SSD_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

LBPH_MODEL_FILE = "lbph_classifier.yml"
FACE_NAMES_FILE = "face_names.pickle"

# setup do Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

_face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=4,
                                  refine_landmarks=False, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
_face_detector_mp = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# carregamento através do SSD
_ssd_net = None
_use_ssd = False
if os.path.exists(SSD_PROTO) and os.path.exists(SSD_MODEL):
    try:
        _ssd_net = cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)
        _use_ssd = True
    except Exception:
        _use_ssd = False

# carregamento através do LBPH
_recognizer = None
_face_names = None
_use_recognizer = False
if os.path.exists(LBPH_MODEL_FILE) and os.path.exists(FACE_NAMES_FILE):
    try:
        _recognizer = cv2.face.LBPHFaceRecognizer_create()
        _recognizer.read(LBPH_MODEL_FILE)
        import pickle
        with open(FACE_NAMES_FILE, "rb") as f:
            _face_names = pickle.load(f)
        
        if isinstance(_face_names, dict):
            vals = list(_face_names.values())
            if all(isinstance(v, int) for v in vals):
                _face_names = {v: k for k, v in _face_names.items()}
        _use_recognizer = True
    except Exception:
        _use_recognizer = False

# enquadramento usando SSD
def detect_faces(frame_bgr, conf_thresh=0.5):
    h, w = frame_bgr.shape[:2]
    boxes = []
    if _use_ssd and _ssd_net is not None:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
                                     (104.0, 117.0, 123.0))
        _ssd_net.setInput(blob)
        dets = _ssd_net.forward()
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf > conf_thresh:
                box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                sx, sy, ex, ey = box.astype("int")
                sx, sy = max(0, sx), max(0, sy)
                ex, ey = min(w - 1, ex), min(h - 1, ey)
                boxes.append((sx, sy, ex, ey))
    else:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = _face_detector_mp.process(rgb)
        if res.detections:
            for det in res.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                boxes.append((max(0, x1), max(0, y1), min(w-1, x1 + bw), min(h-1, y1 + bh)))
    return boxes

# cria landmarks nos rostos registrados
def get_landmarks(frame_bgr, face_box=None):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = _face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    if face_box is None:
        return res.multi_face_landmarks[0]
    sx, sy, ex, ey = face_box
    best = None
    best_iou = 0
    for lm in res.multi_face_landmarks:
        xs = [int(p.x * w) for p in lm.landmark]
        ys = [int(p.y * h) for p in lm.landmark]
        lx1, ly1, lx2, ly2 = min(xs), min(ys), max(xs), max(ys)
        ix1 = max(sx, lx1); iy1 = max(sy, ly1)
        ix2 = min(ex, lx2); iy2 = min(ey, ly2)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        iou = (iw * ih) / ((ex - sx) * (ey - sy) + (lx2 - lx1) * (ly2 - ly1) - iw * ih + 1e-9)
        if iou > best_iou:
            best_iou = iou
            best = lm
    return best

# calcula métricas de um rosto
def compute_pose_and_metrics(frame_bgr, face_box, landmarks=None):
    h, w = frame_bgr.shape[:2]
    sx, sy, ex, ey = face_box
    face_w = ex - sx
    face_h = ey - sy
    center_x = (sx + ex) / 2.0
    center_y = (sy + ey) / 2.0
    rel_face_w = float(face_w) / float(w)

    metrics = {
        "rel_face_width": rel_face_w,
        "center_x": center_x,
        "center_y": center_y,
        "frame_w": w,
        "frame_h": h,
        "brightness": estimate_brightness_from_frame(frame_bgr)
    }

    roll = 0.0; yaw = 0.0; pitch = 0.0
    if landmarks is not None:
        try:
            lm = landmarks.landmark
            def xy(i):
                return int(lm[i].x * w), int(lm[i].y * h)
            left_eye = xy(33)
            right_eye = xy(263)
            nose = xy(1)
            mouth = xy(13)
            
            # roll: espaço entre os olhos
            dy = left_eye[1] - right_eye[1]
            dx = left_eye[0] - right_eye[0]
            roll = float(np.degrees(np.arctan2(dy, dx)))
            if roll > 90:
                roll -= 180
            elif roll < -90:
                roll += 180 
                   
            # yaw heuristic: espaço relativo entre o nariz e o espaço entre os dois olhos
            eyes_cx = (left_eye[0] + right_eye[0]) / 2.0
            yaw = (nose[0] - eyes_cx) / (face_w + 1e-9) * 100.0
            
            # pitch heuristic: y do nariz relativo ao espaço vertical entre os olhos e a boca
            eyes_cy = (left_eye[1] + right_eye[1]) / 2.0
            vertical_span = (mouth[1] - eyes_cy) + 1e-9
            pitch = (nose[1] - eyes_cy) / vertical_span * 50.0
        except Exception:
            pass
    metrics.update({"roll": roll, "yaw": yaw, "pitch": pitch})
    return metrics

# previsor de rosto
def recognize_face_if_possible(face_roi_color, recognizer=None, face_names=None, image_size=(90,120), conf_threshold=80):
    
    if recognizer is None or face_names is None:
        return None, None
    try:
        gray = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2GRAY)
        proc = preprocess_face_for_recognizer(gray, image_size)
        label, conf = recognizer.predict(proc)
        if conf <= conf_threshold:
            name = face_names.get(label, str(label))
            return name, conf
        return None, conf
    except Exception:
        return None, None
