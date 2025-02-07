import torch
import os
import cv2
import mediapipe as mp

from utils.utils import *
from utils.config import MID_X, MID_Y

from .GazeDataLogger import GazeDataLogger
from .GazeModel import GazeModel

class GazeTracker:
    """
    Classe principale pour le suivi du regard.
    Gère l'initialisation du modèle et la prédiction du regard.
    """
    def __init__(self, model_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/checkpoint.pth.tar"))):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_mean()
        self.model = GazeModel()
        self._load_model(model_path)
        self.logger = GazeDataLogger()
        self.margin = 20

    def _load_model(self, model_path):
        """Charge le modèle et ses poids."""
        checkpoint = torch.load(model_path, map_location=self.device,weights_only=True)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_mean(self):
        
        mean_path = os.path.join(os.path.dirname(__file__), "../mat/")
        mean_path = os.path.abspath(mean_path)
        
        self.faceMean = loadMetadata(os.path.join(mean_path, 'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(mean_path, 'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(mean_path, 'mean_right_224.mat'))['image_mean']

        self.faceMean = torch.tensor(self.faceMean / 255.0, dtype=torch.float32)
        self.eyeLeftMean = torch.tensor(self.eyeLeftMean / 255.0, dtype=torch.float32)
        self.eyeRightMean = torch.tensor(self.eyeRightMean / 255.0, dtype=torch.float32)

    def _determine_position(self, pos_x, pos_y):
        """Détermine la position du regard en pixels."""
        if pos_x < MID_X and pos_y < MID_Y:
            return "Top Left"
        elif pos_x > MID_X and pos_y < MID_Y:
            return "Top Right"
        elif pos_x < MID_X and pos_y > MID_Y:
            return "Bottom Left"
        elif pos_x > MID_X and pos_y > MID_Y:
            return "Bottom Right"
        else:
            return "Center"

    def _determine_quadrant(self, gaze_x, gaze_y):
        """Détermine le quadrant en fonction des cm renvoyés par le modèle."""
        if gaze_x < 0 and gaze_y > 0:
            return "Top Left"
        elif gaze_x > 0 and gaze_y > 0:
            return "Top Right"
        elif gaze_x < 0 and gaze_y < 0:
            return "Bottom Left"
        elif gaze_x > 0 and gaze_y < 0:
            return "Bottom Right"
        else:
            return "Center"

    def save_tracking_data(self):
        """Sauvegarde les données du tracking."""
        self.logger.save_data()

    def extract_features(self, img, face_landmarks):
        h, w, _ = img.shape
        
        landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
        
        # Bounbding box
        left_eye_bbox = get_bounding_box(LEFT_EYE, landmarks, x_margin=self.margin, y_margin=self.margin)
        right_eye_bbox = get_bounding_box(RIGHT_EYE, landmarks, x_margin=self.margin, y_margin=self.margin)
        face_bbox = get_bounding_box(FACE_OVAL, landmarks, x_margin=self.margin, y_margin=self.margin)
                        
        # Draw bounding boxes on the image
        cv2.rectangle(img, (left_eye_bbox[0], left_eye_bbox[1]), (left_eye_bbox[2], left_eye_bbox[3]), (0, 255, 0), 2)
        cv2.rectangle(img, (right_eye_bbox[0], right_eye_bbox[1]), (right_eye_bbox[2], right_eye_bbox[3]), (0, 255, 0), 2)
        cv2.rectangle(img, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (255, 0, 0), 2)

        # preprocess region of interests
        left_eye_roi = preprocess_roi(img[left_eye_bbox[1]:left_eye_bbox[3], left_eye_bbox[0]:left_eye_bbox[2]])
        right_eye_roi = preprocess_roi(img[right_eye_bbox[1]:right_eye_bbox[3], right_eye_bbox[0]:right_eye_bbox[2]])       
        face_roi = preprocess_roi(img[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]])
                        
        # generate inputs of model
        left_eye_input = torch.tensor(left_eye_roi, dtype=torch.float32).sub(self.eyeLeftMean).permute(0, 3, 1, 2).to(self.device)
        right_eye_input = torch.tensor(right_eye_roi, dtype=torch.float32).sub(self.eyeRightMean).permute(0, 3, 1, 2).to(self.device)
                                
        face_grid = generate_face_grid(face_bbox, img.shape)
        face_input = torch.tensor(face_roi, dtype=torch.float32).sub(self.faceMean).permute(0, 3, 1, 2).to(self.device)
        face_grid_input = torch.tensor(face_grid, dtype=torch.float32).view(1, -1).to(self.device)
        
        return face_input, left_eye_input, right_eye_input, face_grid_input

    def predict_gaze(self, face_input, left_eye_input, right_eye_input, face_grid_input):
        """
        Effectue la prédiction du regard à partir des entrées traitées.
        Convertit les valeurs en pixels et stocke les résultats.
        """
        
        with torch.no_grad():
            gaze_prediction = self.model(face_input, left_eye_input, right_eye_input, face_grid_input)
            gaze_x, gaze_y = gaze_prediction.cpu().numpy().flatten()

            # Convert in pixels
            pos_x, pos_y = gaze_cm_to_pixels(gaze_x, gaze_y)

            # Determine the position on the screen
            position = self._determine_position(pos_x, pos_y)
            quadrant = self._determine_quadrant(gaze_x, gaze_y)

            # log data
            self.logger.log_data(pos_x, pos_y)

            print(f"Gaze Position (cm): ({gaze_x:.2f}, {gaze_y:.2f}) - {quadrant}, "
                  f"Pixels (x,y): ({pos_x}, {pos_y}) - {position}")

            return pos_x, pos_y, position
    
    def run(self,webcam):
        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
            while True:
                success, img = webcam.read()
                
                if not success:
                    print("Error reading from the webcam.")
                    exit(1)
            
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_mp = face_mesh.process(img_rgb)
                
                if img_mp.multi_face_landmarks:
                    
                    for face_landmarks in img_mp.multi_face_landmarks:
                        
                        face_input, left_eye_input, right_eye_input, face_grid_input = self.extract_features(img, face_landmarks)
                       
                        self.predict_gaze(face_input, left_eye_input, right_eye_input, face_grid_input)
                         
                cv2.imshow("Face Mesh with Gaze Prediction", img)
                
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break
                
        self.logger.save_data()