import os
import cv2
import torch
import pandas as pd
import scipy.io
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio

from torch.utils.data import Dataset, DataLoader

from utils.utils import *

def load_annotation_file(annotation_path):
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split() for line in lines]

def filtered_collate(batch):
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    return {
        'face': torch.stack([item['face'] for item in batch]),
        'eye_left': torch.stack([item['eye_left'] for item in batch]),
        'eye_right': torch.stack([item['eye_right'] for item in batch]),
    }

def generate_means(df, output_dir="../src/mat", batch_size=32, prefix="MPIIFace"):
    dataset = FaceGazeDataset(df)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=filtered_collate)

    sum_face = 0
    sum_eye_left = 0
    sum_eye_right = 0
    count = 0

    for batch in tqdm(dataloader, desc="Computing mean tensors"):
        if batch is None:
            continue
        sum_face += batch['face'].sum(dim=0)          # shape (3, H, W)
        sum_eye_left += batch['eye_left'].sum(dim=0)
        sum_eye_right += batch['eye_right'].sum(dim=0)
        count += batch['face'].shape[0]

    if count == 0:
        print("No valid images to compute means.")
        return

    mean_face = (sum_face / count).numpy()
    mean_eye_left = (sum_eye_left / count).numpy()
    mean_eye_right = (sum_eye_right / count).numpy()

    sio.savemat(f"{output_dir}/mean_face_224_{prefix}.mat", {'mean_face': mean_face})
    sio.savemat(f"{output_dir}/mean_left_224_{prefix}.mat", {'mean_eye_left': mean_eye_left})
    sio.savemat(f"{output_dir}/mean_right_224_{prefix}.mat", {'mean_eye_right': mean_eye_right})

    print("Means saved successfully in:")
    print(f"  → {output_dir}/mean_face_224_{prefix}.mat")
    print(f"  → {output_dir}/mean_left_224_{prefix}.mat")
    print(f"  → {output_dir}/mean_right_224_{prefix}.mat")



class MPIIFaceGazeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for subject in sorted(os.listdir(root_dir)):
            subj_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subj_path):
                continue

            txt_path = os.path.join(subj_path, f"{subject}.txt")
            if not os.path.exists(txt_path):
                continue

            annotations = load_annotation_file(txt_path)

            for ann in annotations:
                img_path = os.path.join(subj_path, ann[0])
                if not os.path.exists(img_path):
                    continue
                self.samples.append((subject, img_path, ann))

    def __len__(self):
        return len(self.samples)

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
                    print(f"Read Error {calib_path}: {e}")

        return pd.DataFrame(records)

    def to_dataframe(self, normalize=True):
        subjects, paths, gaze_xs, gaze_ys = [], [], [], []

        for subject, img_path, ann in self.samples:
            try:
                gaze_x = float(ann[1])
                gaze_y = float(ann[2])

                subjects.append(subject)
                paths.append(img_path)

                gaze_xs.append(gaze_x)
                gaze_ys.append(gaze_y)
            except Exception:
                continue

        df_data = pd.DataFrame({
            'subject': subjects,
            'img_path': paths,
            'gaze_x': gaze_xs,
            'gaze_y': gaze_ys,
        })

        df_screen = self.get_screen_sizes()
        df_merged = df_data.merge(df_screen[['subject', 'width_pixel', 'height_pixel']], on='subject', how='inner')

        if normalize:
            df_merged['gaze_x'] = df_merged['gaze_x'] / df_merged['width_pixel']
            df_merged['gaze_y'] = df_merged['gaze_y'] / df_merged['height_pixel']

        return df_merged[['subject', 'img_path', 'gaze_x', 'gaze_y']]

    def plot_screen_size_histogram(self):
        df = self.get_screen_sizes()
        df['resolution'] = df['width_pixel'].astype(str) + 'x' + df['height_pixel'].astype(str)
        counts = df['resolution'].value_counts().sort_values(ascending=False)

        plt.figure(figsize=(8, 4))
        counts.plot(kind='bar')
        plt.title("Screen resolution distribution (pixels)")
        plt.xlabel("Resolution")
        plt.ylabel("Number of subjects")
        plt.tight_layout()
        plt.show()

class FaceGazeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, means: dict, face_mesh=None):
        self.df = df.reset_index(drop=True)
        self.means = means
        self.face_mesh = face_mesh or mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        img_path = row["img_path"]
        
        gaze = torch.tensor([row["gaze_x"], row["gaze_y"]], dtype=torch.float32)

        features = extract_inputs_from_image(self.face_mesh, img_path, self.means)
        
        if features is None:
            raise RuntimeError(f"Could not extract features from {img_path}")
        
        face, eye_left, eye_right, face_grid = features
        return face, eye_left, eye_right, face_grid, gaze

class FaceGazeBatchDataset(Dataset):
    def __init__(self, pkl_file: str):
        """
        Dataset that loads a single pre-extracted feature batch from a .pkl file.

        :param pkl_file: Path to a single .pkl file containing a list of samples.
        """
        self.pkl_file = pkl_file

        if not os.path.exists(pkl_file):
            raise FileNotFoundError(f"[ERROR] PKL file not found: {pkl_file}")

        try:
            with open(pkl_file, "rb") as f:
                self.samples = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load {pkl_file}: {e}")

        print(f"[OK] Loaded {len(self.samples)} samples from {os.path.basename(pkl_file)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        return sample['face'], sample['eye_left'], sample['eye_right'], sample['face_grid'], torch.tensor(sample['gaze'], dtype=torch.float32)
