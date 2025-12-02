# imports
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# carrega imagens do local de armazenamento
def load_images_from_folder(base_folder, image_size=(90,120)):
    images = []
    labels = []
    label_map = {}
    name_to_id = {}
    current_id = 0

    base = Path(base_folder)
    if not base.exists():
        raise FileNotFoundError(f"Training folder '{base_folder}' not found.")

    for person_dir in sorted(base.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        if name not in name_to_id:
            name_to_id[name] = current_id
            label_map[current_id] = name
            current_id += 1
        label_id = name_to_id[name]
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, image_size)
            images.append(img_resized)
            labels.append(label_id)
    return images, labels, label_map

# treinamento do modelo
def train_from_folder(base_folder, model_out="lbph_classifier.yml", names_out="face_names.pickle", image_size=(90,120)):
    images, labels, label_map = load_images_from_folder(base_folder, image_size=image_size)
    if len(images) == 0:
        raise ValueError("Nenhuma imagem para treinar encontrada.")
    
    # cria o recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # treina o recognizer
    recognizer.train(images, np.array(labels))
    recognizer.write(model_out)
    
    # salva os resultados do treino em um arquico pickle
    with open(names_out, "wb") as f:
        pickle.dump(label_map, f)
    print(f"[TRAIN] Modelo salvo em {model_out}, nomes em {names_out}")

# Se executado diretamente
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train LBPH recognizer from folder.")
    parser.add_argument("--folder", "-f", default="training_images", help="Folder with subfolders per person")
    parser.add_argument("--out", "-o", default="lbph_classifier.yml", help="Output model file")
    parser.add_argument("--names", "-n", default="face_names.pickle", help="Output names pickle")
    parser.add_argument("--w", default=90, type=int)
    parser.add_argument("--h", default=120, type=int)
    args = parser.parse_args()
    train_from_folder(args.folder, model_out=args.out, names_out=args.names, image_size=(args.w, args.h))
