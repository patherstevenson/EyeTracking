import cv2
import os
from tracker.GazeTracker import GazeTracker

def main():
    # Récupérer l'URL de la webcam depuis les variables d'environnement
    webcam_url = os.getenv("WEBCAM_URL", "0")  # Par défaut, "0" pour la webcam locale

    # Si la variable est définie et n'est pas "0", on utilise l'URL
    if webcam_url != "0":
        webcam = cv2.VideoCapture(webcam_url)
    else:
        webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Unable to open webcam. Please check your device or URL.")
        return

    gaze_tracker = GazeTracker()
    
    try:
        gaze_tracker.run(webcam)
    except KeyboardInterrupt:
        print("\nTracking stopped by user.")
    finally:
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
