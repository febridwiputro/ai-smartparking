import numpy as np
import cv2
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew

def deskew(_img):
    image = io.imread(_img)
    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale)
    rotated = rotate(image, angle, resize=True) * 255
    return rotated.astype(np.uint8)

def display_before_after(_original):
    # Load the original image using OpenCV
    original_image = cv2.imread(_original)

    # Deskew the image
    deskewed_image = deskew(_original)

    # Convert deskewed image from RGB to BGR
    deskewed_image_bgr = cv2.cvtColor(deskewed_image, cv2.COLOR_RGB2BGR)

    # Display original and deskewed images using OpenCV
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Deskewed Image', deskewed_image_bgr)
    
    # Wait for a key press and close the image windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Update the path to your image
display_before_after(r'C:\Users\DOT\Documents\febri\github\ai-smartparking\plate_saved\2024-10-23-10-37-51-918009.jpg')


# import cv2
# import numpy as np
# from scipy.ndimage import interpolation as inter

# def correct_skew(image, delta=1, limit=5):
#     def determine_score(arr, angle):
#         data = inter.rotate(arr, angle, reshape=False, order=0)
#         histogram = np.sum(data, axis=1, dtype=float)
#         score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
#         return histogram, score

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

#     scores = []
#     angles = np.arange(-limit, limit + delta, delta)
#     for angle in angles:
#         histogram, score = determine_score(thresh, angle)
#         scores.append(score)

#     best_angle = angles[scores.index(max(scores))]

#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
#     corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
#             borderMode=cv2.BORDER_REPLICATE)

#     return best_angle, corrected

# if __name__ == '__main__':
#     image = cv2.imread(r'C:\Users\DOT\Documents\febri\github\ai-smartparking\plate_saved\2024-10-22-21-41-08-838337.jpg')
#     angle, corrected = correct_skew(image)
#     print('Skew angle:', angle)
#     cv2.imshow('corrected', corrected)
#     cv2.waitKey()

# import math
# from typing import Tuple, Union

# import cv2
# import numpy as np

# from deskew import determine_skew


# def rotate(
#         image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
# ) -> np.ndarray:
#     old_width, old_height = image.shape[:2]
#     angle_radian = math.radians(angle)
#     width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
#     height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

#     image_center = tuple(np.array(image.shape[1::-1]) / 2)
#     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#     rot_mat[1, 2] += (width - old_width) / 2
#     rot_mat[0, 2] += (height - old_height) / 2
#     return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

# image = cv2.imread(r'C:\Users\DOT\Documents\febri\github\ai-smartparking\plate_saved\2024-10-22-21-41-08-838337.jpg')
# grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# angle = determine_skew(grayscale)
# rotated = rotate(image, angle, (0, 0, 0))
# cv2.imwrite(r'C:\Users\DOT\Documents\febri\github\ai-smartparking\plate_saved\2024-10-22-21-41-08-838337_deskew.png', rotated)