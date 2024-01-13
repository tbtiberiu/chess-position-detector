import cv2 as cv

from utils.drawing import draw_lines, draw_points
from utils.intersections_detector import IntersectionsDetector
from utils.lines_detector import LinesDetector
from utils.llr import LLR, llr_pad
from utils.other import resize_image, order_corners, bound_corners, perspective_transform


class ChessboardDetector:
    def __init__(self, image):
        self.original_image = image.copy()
        self.image = resize_image(image)
        self.gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.intersections = []
        self.lines = []
        self.corners = []

    def detect_lines(self):
        self.lines = LinesDetector.detect(self.image)
        return self.lines

    def detect_intersections(self):
        self.intersections = IntersectionsDetector.detect(self.image, self.lines)
        return self.intersections

    def detect_corners(self):
        self.corners = LLR(self.image, self.intersections, self.lines)
        self.corners = llr_pad(self.corners)
        self.corners = order_corners(self.corners)
        self.corners = bound_corners(self.corners, self.image.shape[1], self.image.shape[0])
        return self.corners

    def transform(self):
        self.corners = order_corners(self.corners)
        self.image = perspective_transform(self.image, self.corners)
        self.gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        return self.image

    def detect(self):
        self.detect_lines()
        self.detect_intersections()
        self.detect_corners()
        return self.image

    def transform_and_detect(self):
        self.transform()
        self.detect_lines()
        self.detect_intersections()
        self.detect_corners()
        return self.image

    def draw(self):
        image = self.image.copy()
        draw_lines(image, self.lines)
        draw_points(image, self.intersections)
        draw_points(image, self.corners, color=(0, 0, 255))
        return image
