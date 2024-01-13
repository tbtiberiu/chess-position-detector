import cv2 as cv

def draw_lines(img, lines, color=(0, 0, 255), thickness=1):
    """Draw lines on an image"""
    for line in lines:
        x1, y1, x2, y2 = line[0]
        img = cv.line(img, (x1, y1), (x2, y2), color, thickness)
    return img

def draw_points(img, points, color=(255, 0, 0), radius=5):
    """Draw points on an image"""
    for point in points:
        img = cv.circle(img, point, radius, color, -1)
    return img