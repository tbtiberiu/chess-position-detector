from utils.drawing import draw_lines, draw_points
from utils.intersections_detector import IntersectionsDetector
from utils.lines_detector import LinesDetector
from utils.llr import LLR, llr_pad
from utils.other import resize_image, order_corners, bound_corners, perspective_transform, point_transform


class ChessboardDetector:
    LAYER_COUNT = 3

    def __init__(self):
        self.original_image = None
        self.image = None
        self.intersections = []
        self.lines = []
        self.corners = []
        self.transform_matrices = []

    def set_image(self, image):
        self.original_image = image.copy()
        self.image = resize_image(image)
        self.intersections = []
        self.lines = []
        self.corners = []
        self.transform_matrices = []
        return self.image

    def detect_lines(self):
        self.lines = LinesDetector.detect(self.image)
        return self.lines

    def detect_intersections(self):
        self.intersections = IntersectionsDetector.detect(self.image, self.lines)
        return self.intersections

    def detect_corners(self):
        self.corners = LLR(self.image, self.intersections, self.lines)
        self.corners = llr_pad(self.corners)
        self.corners = bound_corners(self.corners, self.image.shape[1], self.image.shape[0])
        self.corners = order_corners(self.corners)
        return self.corners

    def transform(self):
        self.image, transform_matrix = perspective_transform(self.image, self.corners)
        self.transform_matrices.append(transform_matrix)
        return self.image

    def transform_point(self, point):
        return point_transform(point, self.transform_matrices)

    def detect_components(self):
        self.detect_lines()
        self.detect_intersections()
        if len(self.intersections) < 4:
            return self.lines, self.intersections, self.corners
        self.detect_corners()
        return self.lines, self.intersections, self.corners

    def layer(self):
        self.detect_lines()
        self.detect_intersections()
        if len(self.intersections) < 4:
            return
        self.detect_corners()
        self.transform()

    def detect(self, image, layer_count=LAYER_COUNT):
        self.set_image(image)
        for _ in range(layer_count):
            self.layer()
        return self.image

    def draw(self):
        image = self.image.copy()
        draw_lines(image, self.lines)
        draw_points(image, self.intersections)
        draw_points(image, self.corners, color=(0, 0, 255))
        return image
