from ultralytics import YOLO

class ChessPiecesDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image, save=True)
        return results