import os
import cv2
import torch
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


def load_annotation_file(annotation_path):
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split() for line in lines]


def create_face_grid(x, y, w, h, grid_size=25, image_w=640, image_h=480):
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    cell_w, cell_h = image_w / grid_size, image_h / grid_size

    left = int(x / cell_w)
    top = int(y / cell_h)
    right = int((x + w) / cell_w)
    bottom = int((y + h) / cell_h)

    grid[top:bottom + 1, left:right + 1] = 1.0
    return grid


class MPIIGazeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for subject in sorted(os.listdir(root_dir)):
            subj_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subj_path):
                continue

            for day in sorted(os.listdir(subj_path)):
                day_path = os.path.join(subj_path, day)
                ann_path = os.path.join(day_path, 'annotation.txt')
                img_dir = day_path  # Les images sont directement dans le dossier dayXX

                if not os.path.exists(ann_path):
                    continue

                annotations = load_annotation_file(ann_path)

                for idx, values in enumerate(annotations):
                    img_path = os.path.join(img_dir, f"{idx+1:04}.jpg")
                    if not os.path.exists(img_path):
                        continue
                    self.samples.append((img_path, values))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Dans MPIIGaze Original : les images sont des patches yeux gauche + droit collés
        # donc on divise l'image en deux parties
        h, w, _ = img.shape
        left_eye = img[:, :w // 2]
        right_eye = img[:, w // 2:]

        # Fake face = concaténation des deux yeux (juste pour respecter l'entrée du modèle)
        face_crop = img  # on peut aussi faire une moyenne ou un padding

        transform = self.transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        face_tensor = transform(face_crop)
        left_tensor = transform(left_eye)
        right_tensor = transform(right_eye)

        # Grille faciale neutre (pas de bbox dans ce cas)
        face_grid = torch.zeros((1, 25, 25), dtype=torch.float32)

        # Labels : coordonnées écran 2D (on-screen gaze target)
        gaze_x = float(ann[24])
        gaze_y = float(ann[25])
        label = torch.tensor([gaze_x, gaze_y], dtype=torch.float32)

        return {
            'face': face_tensor,
            'eye_left': left_tensor,
            'eye_right': right_tensor,
            'face_grid': face_grid,
            'label': label
        }

    def to_dataframe(self):
        img_paths, gaze_xs, gaze_ys = [], [], []
        for img_path, ann in self.samples:
            if len(ann) > 25:
                try:
                    gaze_x = float(ann[24])
                    gaze_y = float(ann[25])
                    img_paths.append(img_path)
                    gaze_xs.append(gaze_x)
                    gaze_ys.append(gaze_y)
                except ValueError:
                    continue
        return pd.DataFrame({
            'img_path': img_paths,
            'gaze_x': gaze_xs,
            'gaze_y': gaze_ys
        })

    def get_screen_sizes(self):
        records = []
        for subject in sorted(os.listdir(self.root_dir)):
            subj_path = os.path.join(self.root_dir, subject)
            if not os.path.isdir(subj_path):
                continue

            calib_path = os.path.join(subj_path, 'Calibration', 'screenSize.mat')
            if os.path.exists(calib_path):
                try:
                    mat = scipy.io.loadmat(calib_path)
                    record = {
                        'subject': subject,
                        'width_pixel': int(mat['width_pixel'][0][0]),
                        'height_pixel': int(mat['height_pixel'][0][0]),
                        'width_mm': float(mat['width_mm'][0][0]),
                        'height_mm': float(mat['height_mm'][0][0])
                    }
                    records.append(record)
                except Exception as e:
                    print(f"Erreur lecture {calib_path}: {e}")

        return pd.DataFrame(records)

    def plot_screen_size_histogram(self):
        df = self.get_screen_sizes()
        df['resolution'] = df['width_pixel'].astype(str) + 'x' + df['height_pixel'].astype(str)
        counts = df['resolution'].value_counts().sort_values(ascending=False)

        plt.figure(figsize=(8, 4))
        counts.plot(kind='bar')
        plt.title("Distribution des résolutions d'écran (pixels)")
        plt.xlabel("Résolution")
        plt.ylabel("Nombre de sujets")
        plt.tight_layout()
        plt.show()
