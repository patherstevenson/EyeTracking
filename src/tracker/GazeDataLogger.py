import os
import pickle
import datetime

class GazeDataLogger:
    """
    Classe responsable de l'enregistrement et de la sauvegarde des données de suivi du regard.
    """

    def __init__(self):
        self.tracking_data = []
        self.tracking_folder = "src/experiments/"
        os.makedirs(self.tracking_folder, exist_ok=True)

        # Définir un nom de fichier unique basé sur la date et l'heure
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.tracking_file = os.path.join(self.tracking_folder, f"gaze_{self.timestamp}.pkl")

    def log_data(self, pos_x, pos_y):
        """Ajoute une nouvelle entrée au fichier de suivi."""
        self.tracking_data.append((pos_x, pos_y))

    def save_data(self):
        """Sauvegarde les données enregistrées dans un fichier pickle."""
        with open(self.tracking_file, "wb") as f:
            pickle.dump(self.tracking_data, f)
        print(f"Tracking data saved to {self.tracking_file}")
