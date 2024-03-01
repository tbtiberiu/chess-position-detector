import math
import cv2 as cv
import numpy as np

from statistics import median


def distance(point1, point2):
    """Calculate distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def median_distance(points, num_neighbors=8):
    """Calculate median distance between points"""
    distances = []

    for i in range(len(points)):
        all_distances = []
        for j in range(len(points)):
            if i != j:
                dist = distance(points[i], points[j])
                all_distances.append(dist)

        # Sort distances and consider the closest num_neighbors distances
        all_distances.sort()
        closest_distances = all_distances[:num_neighbors]
        avg_closest_dist = sum(closest_distances) / num_neighbors
        distances.append(avg_closest_dist)

    return int(median(distances))

def resize_image(image, max_size=700):
    """Resize image to max_size"""
    height, width = image.shape[:2]
    min_image_size = min(height, width)

    if min_image_size > max_size:
        if width > height:
            new_width = max_size
            new_height = int(new_width * (height / width))
        else:
            new_height = max_size
            new_width = int(new_height * (width / height))
        image = cv.resize(image, (new_width, new_height))

    return image

def order_corners(points):
    """Order corners from points"""
    if not points or len(points) < 4:
        return "Insufficient points to form a shape"

    top_left = points[0]
    top_right = points[0]
    bottom_left = points[0]
    bottom_right = points[0]
    min_sum = points[0][0] + points[0][1]
    max_sum = points[0][0] + points[0][1]
    min_diff = points[0][0] - points[0][1]
    max_diff = points[0][0] - points[0][1]

    for point in points:
        points_sum = point[0] + point[1]
        points_diff = point[0] - point[1]

        if points_sum < min_sum:
            min_sum = points_sum
            top_left = point
        if points_sum > max_sum:
            max_sum = points_sum
            bottom_right = point

        if points_diff < min_diff:
            min_diff = points_diff
            bottom_left = point
        if points_diff > max_diff:
            max_diff = points_diff
            top_right = point

    return top_left, bottom_left, top_right, bottom_right

def bound_corners(points, image_width, image_height):
    """Bound corners to image size"""
    for i in range(4):
        if points[i][0] < 0:
            points[i][0] = 0
        if points[i][1] < 0:
            points[i][1] = 0
        if points[i][0] > image_width:
            points[i][0] = image_width
        if points[i][1] > image_height:
            points[i][1] = image_height

    return points

def perspective_transform(image, corner_points):
    """Perform perspective transform"""
    width, height = corner_points[3][0] - corner_points[0][0], corner_points[1][1] - corner_points[0][1]
    output_size = (width, height)
    output_points = np.array([[0, 0], [0, output_size[1]], [output_size[0], 0], [output_size[0], output_size[1]]], dtype=np.float32)

    corner_points = np.array(corner_points, dtype=np.float32)

    matrix = cv.getPerspectiveTransform(corner_points, output_points)
    transformed_image = cv.warpPerspective(image, matrix, output_size)

    return transformed_image, matrix

def point_transform(point, transform_matrices):
    """Transform point using matrices"""
    transformed_point = np.array([point], dtype=np.float32)

    for transform_matrix in transform_matrices:
        transformed_point = cv.perspectiveTransform(transformed_point, transform_matrix)

    return transformed_point[0][0]