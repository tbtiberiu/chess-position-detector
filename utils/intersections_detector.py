import cv2 as cv
import numpy as np

from keras.models import load_model
from sklearn.cluster import DBSCAN
from utils.other import median_distance


class IntersectionsDetector:
    @staticmethod
    def get_intersections(lines):
        """Find intersections between lines"""
        intersections = []
        lines_count = len(lines)

        for i in range(lines_count):
            for j in range(i + 1, lines_count):
                line1 = lines[i][0]
                line2 = lines[j][0]
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2
                denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))

                if denominator != 0:  # To avoid division by zero
                    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
                    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

                    if 0 <= ua <= 1 and 0 <= ub <= 1:
                        intersection_x = x1 + ua * (x2 - x1)
                        intersection_y = y1 + ua * (y2 - y1)
                        intersections.append((int(intersection_x), int(intersection_y)))
        return intersections

    @staticmethod
    def cluster_points(points, max_distance=3):
        """Cluster points using DBSCAN"""
        if len(points) == 0:
            return []

        points_array = np.array(points)

        dbscan = DBSCAN(eps=max_distance, min_samples=1)
        clusters = dbscan.fit_predict(points_array)

        clustered_points = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in clustered_points:
                clustered_points[cluster_id] = []
            clustered_points[cluster_id].append(points[i])

        mean_points = []
        for cluster_id in clustered_points:
            mean_point = np.mean(clustered_points[cluster_id], axis=0, dtype=np.int32)
            mean_points.append(tuple(mean_point))

        return mean_points

    @staticmethod
    def remove_outliers(points, max_distance, alpha=20):
        """Remove outliers from points"""
        if len(points) == 0:
            return []

        points_array = np.array(points)

        dbscan = DBSCAN(eps=max_distance+alpha, min_samples=5)
        labels = dbscan.fit_predict(points_array)

        # Find the largest cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts)]

        # Get points belonging to the largest cluster
        points_in_largest_cluster = points_array[labels == largest_cluster_label]

        filtered_points = []
        for point in points_in_largest_cluster:
            filtered_points.append(tuple(point))

        return filtered_points

    @staticmethod
    def filter_intersections(image, intersections, size=10):
        """Filter intersections to remove false positives"""
        model = load_model('models/lattice_points.model.keras')

        filtered_intersections = []

        if len(image.shape) == 3:
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray_image = image

        for intersection in intersections:
            lx1 = max(0, int(intersection[0] - size - 1))
            lx2 = max(0, int(intersection[0] + size))
            ly1 = max(0, int(intersection[1] - size))
            ly2 = max(0, int(intersection[1] + size + 1))

            dimg = gray_image[ly1:ly2, lx1:lx2]

            if dimg.shape[0] <= 0 or dimg.shape[1] <= 0: continue

            dimg = cv.threshold(dimg, 0, 255, cv.THRESH_OTSU)[1]
            dimg = cv.Canny(dimg, 0, 255)
            dimg = cv.resize(dimg, (21, 21), interpolation=cv.INTER_CUBIC)

            X = dimg.reshape(1, 21, 21, 1) / 255.0

            predict_x = model(X)
            classes_x = np.argmax(predict_x, axis=1)
            prediction, confidence = classes_x[0], predict_x[0][classes_x[0]]

            if not prediction: continue
            if prediction == 0 and confidence >= 0.95: continue
            if intersection[0] < 0 or intersection[1] < 0: continue

            filtered_intersections.append(intersection)

        return filtered_intersections

    @staticmethod
    def detect(image, lines):
        """Detect intersections between lines"""
        intersections = IntersectionsDetector.get_intersections(lines)
        if len(intersections) < 4:
            return []
        intersections = IntersectionsDetector.cluster_points(intersections)
        intersections = IntersectionsDetector.filter_intersections(image, intersections)
        if len(intersections) < 4:
            return []
        intersections = IntersectionsDetector.cluster_points(intersections, 15)
        if len(intersections) < 4:
            return []
        intersections = IntersectionsDetector.remove_outliers(intersections, median_distance(intersections))
        return intersections
