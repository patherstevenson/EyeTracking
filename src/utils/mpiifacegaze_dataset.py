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
from torchvision import transforms

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

    def __getitem__(self, idx):
        subject, img_path, ann = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transform = self.transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        face_tensor = transform(img)

        gaze_x = float(ann[1])
        gaze_y = float(ann[2])
        label = torch.tensor([gaze_x, gaze_y], dtype=torch.float32)
        face_grid = torch.zeros((1, 25, 25), dtype=torch.float32)

        return {
            'face': face_tensor,
            'face_grid': face_grid,
            'label': label,
            'subject': subject,
            'eye_used': ann[-1]
        }

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
    def __init__(self, dataframe, eye_crop_size=(64, 64), face_size=(224, 224), grid_size=25):
        self.data = dataframe
        self.eye_crop_size = eye_crop_size
        self.face_size = face_size
        self.grid_size = grid_size

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['img_path']
        gaze_x, gaze_y = row['gaze_x'], row['gaze_y']

        img = cv2.imread(img_path)
        if img is None:
            with open("skipped_images.txt", "a") as f:
                f.write(f"[NOT FOUND] {img_path}\n")
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_results = self.face_mesh.process(img_rgb)

        if not mp_results.multi_face_landmarks:
            with open("skipped_images.txt", "a") as f:
                f.write(f"[NO FACE] {img_path}\n")
            return None

        landmarks = mp_results.multi_face_landmarks[0]

        h, w = img.shape[:2]
        points = [(int(pt.x * w), int(pt.y * h)) for pt in landmarks.landmark]

        try:
            left_eye_bbox = get_bounding_box(LEFT_EYE, points, w, h)
            right_eye_bbox = get_bounding_box(RIGHT_EYE, points, w, h)
            face_bbox = get_bounding_box(FACE_OVAL, points, w, h)

            if not all(map(is_valid, [left_eye_bbox, right_eye_bbox, face_bbox])):
                with open("skipped_images.txt", "a") as f:
                    f.write(f"[INVALID BBOX] {img_path}\n")
                return None

            left_eye_roi = preprocess_roi(img[left_eye_bbox[1]:left_eye_bbox[3], left_eye_bbox[0]:left_eye_bbox[2]])
            right_eye_roi = preprocess_roi(img[right_eye_bbox[1]:right_eye_bbox[3], right_eye_bbox[0]:right_eye_bbox[2]])
            face_roi = preprocess_roi(img[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]])

        except Exception as e:
            with open("skipped_images.txt", "a") as f:
                f.write(f"[EXCEPTION] {img_path} -- {str(e)}\n")
            return None

        eye_left_tensor = torch.tensor(left_eye_roi[0], dtype=torch.float32).permute(2, 0, 1)
        eye_right_tensor = torch.tensor(right_eye_roi[0], dtype=torch.float32).permute(2, 0, 1)
        face_tensor = torch.tensor(face_roi[0], dtype=torch.float32).permute(2, 0, 1)

        face_grid = generate_face_grid(face_bbox, img.shape)
        face_grid_tensor = torch.tensor(face_grid, dtype=torch.float32).view(1, -1)

        gaze = torch.tensor([gaze_x, gaze_y], dtype=torch.float32)

        return {
            'face': face_tensor,
            'eye_left': eye_left_tensor,
            'eye_right': eye_right_tensor,
            'face_grid': face_grid_tensor,
            'gaze': gaze
        }
