from ultralytics import YOLO

class ChessPiecesDetector:
    MODEL_PATH = 'models/chess_pieces.model.pt'

    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image, verbose=False)
        return results