import cv2 as cv
import numpy as np

from utils.hough_bundler import HoughBundler

CLAHE_PARAMS = [
    {'limit': 3, 'grid': (2, 6), 'iterations': 5},
    {'limit': 3, 'grid': (6, 2), 'iterations': 5},
    {'limit': 5, 'grid': (3, 3), 'iterations': 5},
    {'limit': 0, 'grid': (0, 0), 'iterations': 0},
]


class LinesDetector:
    @staticmethod
    def canny(img, alpha=0.3):
        """Canny Edge Detector"""
        median = np.median(img)
        img = cv.bilateralFilter(img, 7, 75, 75)
        t1 = int(max(0, (1.0 - alpha) * median))
        t2 = int(min(255, (1.0 + alpha) * median))
        return cv.Canny(img, t1, t2)

    @staticmethod
    def hough_lines(img, threshold=40, minLineLength=40, maxLineGap=70):
        """Hough Line Detector"""
        lines = cv.HoughLinesP(img, rho=1, theta=np.pi/180,
                               threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
        if lines is None:
            return []
        return lines

    @staticmethod
    def clahe(img, limit=2, grid=(3, 3), iterations=5):
        """Adaptive Histogram Equalization (CLAHE)"""
        for _ in range(iterations):
            img = cv.createCLAHE(clipLimit=limit, tileGridSize=grid).apply(img)
        if limit != 0:
            kernel = np.ones((10, 10), np.uint8)
            img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        return img

    @staticmethod
    def all_clahe_lines(img, clahe_params=CLAHE_PARAMS):
        """Find lines in an image using CLAHE, Canny Edge Detector and Hough Line Detector"""
        results = []
        for params in clahe_params:
            clahe_image = LinesDetector.clahe(img, params['limit'], params['grid'], params['iterations'])
            edges = LinesDetector.canny(clahe_image)
            lines = LinesDetector.hough_lines(edges)
            results += list(lines)
        return results

    @staticmethod
    def filter_lines(lines, min_length=50, angle_threshold=30):
        """Filter lines by angle, length and gap"""
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.arctan2(y2 - y1, x2 - x1)

            while angle < 0.0:
                angle += np.pi

            angle_degrees = np.degrees(angle)
            if min_length < length:
                if angle_degrees < angle_threshold or angle_degrees > 180 - angle_threshold:
                    filtered_lines.append(line)  # horizontal
                elif angle_degrees < 90 + angle_threshold and angle_degrees > 90 - angle_threshold:
                    filtered_lines.append(line)  # vertical
        return filtered_lines

    @staticmethod
    def detect(image):
        """Get straight lines from an image"""
        if image is None:
            return []

        if len(image.shape) == 3:
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray_image = image

        lines = LinesDetector.all_clahe_lines(gray_image)
        lines = LinesDetector.filter_lines(lines)
        bundler = HoughBundler(min_distance=10, min_angle=0.5)
        lines = bundler.process_lines(lines)
        return lines
