from ultralytics import YOLO


class ChessPiecesDetector:
    MODEL_PATH = 'models/chess_pieces.model.pt'

    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image, verbose=False)
        return results

    def detect_piece_class(self, piece_image):
        piece_results = self.model.predict(piece_image, verbose=False)
        for piece_box in piece_results[0].boxes:
            return piece_box.cls
        return None