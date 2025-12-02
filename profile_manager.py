# imports
import os
from helper_functions import save_json, load_json, ensure_dir
import numpy as np

# permanência de dados de perfil
PROFILE_FILE = "profile_data.json"

class ProfileManager:
    def __init__(self, profile_path=PROFILE_FILE):
        self.profile_path = profile_path
        self.profile = load_json(profile_path) or {}

    # salvar métricas atuais em um perfil
    def save_profile_from_metrics(self, name, metrics: dict):
        """
        metrics = {
          "rel_face_width": 0.33,
          "center_x": 320,
          "center_y": 240,
          "brightness": 140,
          "roll": 0.0,
          "yaw": 0.0,
          "pitch": 0.0
        }
        """
        ensure_dir(os.path.dirname(self.profile_path) or ".")
        self.profile[name] = metrics
        save_json(self.profile_path, self.profile)

    def get_profile(self, name):
        return self.profile.get(name)

    def list_profiles(self):
        return list(self.profile.keys())

    # ver se tem algum perfil existente com métricas similares
    def best_match(self, metrics, threshold=0.2):
        if not self.profile:
            return None, None
        best_name = None
        best_score = float("inf")
        for name, m in self.profile.items():
            if not isinstance(metrics, dict):
                return None, 1e9
            if not isinstance(m, dict):
                continue

            d_width = abs(m.get("rel_face_width", 0) - metrics.get("rel_face_width", 0))
            d_bright = abs(m.get("brightness", 0) - metrics.get("brightness", 0))
            
            score = d_width + d_bright * 0.3
            if score < best_score:
                best_score = score
                best_name = name
        if best_score <= threshold:
            return best_name, best_score
        return None, best_score

    def save_to_disk(self):
        save_json(self.profile_path, self.profile)
