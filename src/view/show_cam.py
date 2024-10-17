import cv2
import numpy as np


def show_cam(text, image, max_width=1080, max_height=720):
    res_img = resize_image(image, max_width, max_height)
    cv2.imshow(text, res_img)
    # cv2.imshow(text, image)

def show_text(text, image, x_shape, y_shape, color=(255, 255, 255)):
    image = cv2.rectangle(image, (x_shape - 10, y_shape + 20), (x_shape + 400, y_shape - 40), (0, 0, 0), -1)
    return cv2.putText(image, text, (x_shape, y_shape), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scale = max_width / width if width > height else max_height / height
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return image

def show_box(image, box, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

def show_polygon(image, points: list[list[int]], color=(0, 255, 0)):
    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
    return image

def show_line(frame, point1, point2):
    cv2.line(frame, point1, point2, (0, 255, 0), 2)
    return frame


# def save_cam(folder_name, frame):

