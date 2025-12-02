# imports
import os
import sys
import time
import threading
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# módulos locais
from helper_functions import ensure_dir, estimate_brightness_from_frame, preprocess_face_for_recognizer
from face_analyzer import detect_faces, get_landmarks, compute_pose_and_metrics, recognize_face_if_possible
from profile_manager import ProfileManager

# objetos de reconhecimento opcional
import face_analyzer as FA

# métricas
CONFIG = {
    "video_w": 640,
    "video_h": 480,
    "center_tol": 0.05,
    "face_width_tol_min": 0.15,
    "face_width_tol_max": 0.50,
    "roll_tol_deg": 25,
    "yaw_tol_deg": 15,
    "pitch_tol_deg": 30,
    "recognizer_conf_threshold": 80,
    "capture_images_per_profile": 8,
    "profile_image_size": (90, 120)
}

# permanência das imagens
TRAINING_FOLDER = "training_images"

# aplicação
class FrameHelperApp:
    def __init__(self, root):
        self.root = root
        root.title("Frame Helper")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir a webcam.")
            sys.exit(1)

        self.video_w = CONFIG["video_w"]
        self.video_h = CONFIG["video_h"]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_h)

        # UI
        self.canvas = tk.Canvas(root, width=self.video_w, height=self.video_h, bg="black")
        self.canvas.pack(padx=8, pady=8)

        self.tip_var = tk.StringVar(value="Aguardando início")
        self.tip_label = ttk.Label(root, textvariable=self.tip_var, wraplength=self.video_w-20)
        self.tip_label.pack(pady=(0, 8))

        btn_frame = ttk.Frame(root)
        btn_frame.pack()

        self.start_btn = ttk.Button(btn_frame, text="Iniciar", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=6)
        self.stop_btn = ttk.Button(btn_frame, text="Parar", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=6)
        self.create_profile_btn = ttk.Button(btn_frame, text="Salvar perfil (rápido)", command=self.create_profile)
        self.create_profile_btn.grid(row=0, column=2, padx=6)

        self.profile_manager = ProfileManager()
        self.recent_tips = deque(maxlen=6)
        self._running = False

    # loop de captura
    def start(self):
        if not self._running:
            self._running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.tip_var.set("Câmera iniciada...")
            self.update_frame()

    def stop(self):
        if self._running:
            self._running = False
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.tip_var.set("Parado.")

    def update_frame(self):
        if not self._running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.tip_var.set("Falha na captura da câmera.")
            self.root.after(20, self.update_frame)
            return

        frame = cv2.resize(frame, (self.video_w, self.video_h))

        out_frame, tip = self.process_frame(frame)

        img = Image.fromarray(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(img)
        self.canvas.photo = photo
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)

        self.recent_tips.append(tip)
        try:
            tip_text = max(set(self.recent_tips), key=self.recent_tips.count)
        except Exception:
            tip_text = tip

        self.tip_var.set(tip_text)

        self.root.after(20, self.update_frame)

    # processamento de frames e mensagens
    def process_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        tips = []

        bright = estimate_brightness_from_frame(frame_bgr)
        if bright < 50:
            tips.append("Iluminação fraca — aproxime-se de uma fonte de luz.")
        elif bright > 230:
            tips.append("Luz muito forte — evite luz direta atrás de você.")

        boxes = detect_faces(frame_bgr)
        if not boxes:
            return frame_bgr, "Nenhum rosto detectado — enquadre seu rosto na câmera."

        main_box = boxes[0]
        sx, sy, ex, ey = main_box
        cv2.rectangle(frame_bgr, (sx, sy), (ex, ey), (0, 200, 0), 2)

        landmarks = get_landmarks(frame_bgr, main_box)
        metrics = compute_pose_and_metrics(frame_bgr, main_box, landmarks)

        dx = (metrics["center_x"] / w) - 0.5
        dy = (metrics["center_y"] / h) - 0.5

        if abs(dx) > CONFIG["center_tol"]:
            tips.append("Mova-se para a esquerda" if dx > 0 else "Mova-se para a direita")
        if abs(dy) > CONFIG["center_tol"]:
            tips.append("Abaixe a câmera" if dy > 0 else "Levante a câmera")

        if metrics["rel_face_width"] < CONFIG["face_width_tol_min"]:
            tips.append("Aproxime-se um pouco")
        elif metrics["rel_face_width"] > CONFIG["face_width_tol_max"]:
            tips.append("Afaste-se um pouco")

        if abs(metrics.get("roll", 0)) > CONFIG["roll_tol_deg"]:
            tips.append("Incline levemente a cabeça para nivelar")

        if abs(metrics.get("yaw", 0)) > CONFIG["yaw_tol_deg"]:
            tips.append("Vire levemente a cabeça para centralizar")

        if abs(metrics.get("pitch", 0)) > CONFIG["pitch_tol_deg"]:
            tips.append("Ajuste seu olhar (olhe um pouco para cima ou para baixo)")

        profile_name, score = self.profile_manager.best_match({
            "rel_face_width": metrics["rel_face_width"],
            "brightness": metrics["brightness"]
        }, threshold=0.3)

        if profile_name:
            tips.append(f"Perfil detectado: {profile_name}")

        unique = list(dict.fromkeys(tips))
        tip_text = " · ".join(unique) if unique else "Enquadramento ideal."

        return frame_bgr, tip_text

    # criação de perfis de usuário
    def create_profile(self):
        messagebox.showinfo("Perfil", "Função completa de perfis ainda não implementada nesta versão.")

    def on_close(self):
        self.stop()
        try:
            self.cap.release()
        except:
            pass
        self.root.destroy()

# inicialização
def main():
    root = tk.Tk()
    app = FrameHelperApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.geometry("700x620")
    root.mainloop()

if __name__ == "__main__":
    main()
