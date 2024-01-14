from detectors.chess_pieces_detector import ChessPiecesDetector
from detectors.chessboard_detector import ChessboardDetector

class ChessPositionDetector:
    def __init__(self):
        self.chessboard_detector = ChessboardDetector()
        self.chess_pieces_detector = ChessPiecesDetector()

    def detect(self, image):
        chessboard_image = self.chessboard_detector.detect(image)
        chess_pieces_result = self.chess_pieces_detector.detect(chessboard_image)[0]

        image_width = chess_pieces_result.orig_shape[1]
        image_height = chess_pieces_result.orig_shape[0]

        box_width = image_width / 8
        box_height = image_height / 8

        chess_pieces = ['.', 'b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']
        piece_positions = []
        for box in chess_pieces_result.boxes:
            piece_type = chess_pieces[int(box.cls)]
            xmin, ymin, xmax, ymax = box.xyxy[0]
            x_middle = (xmin + xmax) / 2
            y_middle = ymax - (box_height / 2)

            col = int(x_middle / box_width)
            row = int(y_middle / box_height)
            piece_positions.append((piece_type, (row, col)))

        piece_positions.sort(key=lambda x: (x[1][0], x[1][1]))

        chessboard = [['.' for _ in range(8)] for _ in range(8)]

        for piece_type, (row, col) in piece_positions:
            chessboard[row][col] = piece_type

        fen_rows = []
        for row in chessboard:
            fen_row = ''
            empty_count = 0
            for square in row:
                if square == '.':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += square
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)

        fen = '/'.join(fen_rows)

        return fen