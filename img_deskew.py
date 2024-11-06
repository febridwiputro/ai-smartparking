import cv2
import numpy as np
import matplotlib.pyplot as plt

def shi_tomashi(image):
    """
    Use Shi-Tomashi algorithm to detect corners

    Args:
        image: np.array

    Returns:
        corners: list

    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 100)
    corners = np.int0(corners)
    corners = sorted(np.concatenate(corners).tolist())
    print('\nThe corner points are...\n')

    im = image.copy()
    for index, c in enumerate(corners):
        x, y = c
        cv2.circle(im, (x, y), 3, 255, -1)
        character = chr(65 + index)
        print(character, ':', c)
        cv2.putText(im, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    plt.imshow(im)
    plt.title('Corner Detection: Shi-Tomashi')
    plt.show()
    return corners


def get_destination_points(corners):
    """
    -Get destination points from corners of warped images
    -Approximating height and width of the rectangle: we take maximum of the 2 widths and 2 heights

    Args:
        corners: list

    Returns:
        destination_corners: list
        height: int
        width: int

    """

    w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2)
    h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])

    print('\nThe destination points are: \n')
    for index, c in enumerate(destination_corners):
        character = chr(65 + index) + "'"
        print(character, ':', c)

    print('\nThe approximated height and width of the original image is: \n', (h, w))
    return destination_corners, h, w


def unwarp(img, src, dst):
    """

    Args:
        img: np.array
        src: list
        dst: list

    Returns:
        un_warped: np.array

    """
    h, w = img.shape[:2]
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    print('\nThe homography matrix is: \n', H)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)

    # plot

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    # f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(img)
    ax1.set_title('Original Image')

    x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
    y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]

    ax2.imshow(img)
    ax2.plot(x, y, color='yellow', linewidth=3)
    ax2.set_ylim([h, 0])
    ax2.set_xlim([0, w])
    ax2.set_title('Target Area')

    plt.show()
    return un_warped


def example_one():
    """
    Skew correction using homography and corner detection using Shi-Tomashi corner detector

    Returns: None

    """
    image = cv2.imread(r"D:\engine\cv\image_restoration\backup\18-48\plate_saved\2024-10-24-18-32-55-343434.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title('Original Image')
    plt.show()

    corners = shi_tomashi(image)
    destination, h, w = get_destination_points(corners)
    un_warped = unwarp(image, np.float32(corners), destination)
    cropped = un_warped[0:h, 0:w]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), facecolor='w', edgecolor='k')
    # f.subplots_adjust(hspace=.2, wspace=.05)

    ax1.imshow(un_warped)
    ax2.imshow(cropped)

    plt.show()


def apply_filter(image):
    """
    Define a 5X5 kernel and apply the filter to gray scale image
    Args:
        image: np.array

    Returns:
        filtered: np.array

    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.float32) / 15
    filtered = cv2.filter2D(gray, -1, kernel)
    plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
    plt.title('Filtered Image')
    plt.show()
    return filtered

def apply_threshold(filtered):
    """
    Apply OTSU threshold

    Args:
        filtered: np.array

    Returns:
        thresh: np.array

    """
    ret, thresh = cv2.threshold(filtered, 250, 255, cv2.THRESH_OTSU)
    plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
    plt.title('After applying OTSU threshold')
    plt.show()
    return thresh

def detect_contour(img, image_shape):
    """

    Args:
        img: np.array()
        image_shape: tuple

    Returns:
        canvas: np.array()
        cnt: list

    """
    canvas = np.zeros(image_shape, np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(canvas, cnt, -1, (0, 255, 255), 3)
    plt.title('Largest Contour')
    plt.imshow(canvas)
    plt.show()

    return canvas, cnt

def detect_corners_from_contour(canvas, cnt):
    """
    Detecting corner points form contours using cv2.approxPolyDP()
    Args:
        canvas: np.array()
        cnt: list

    Returns:
        approx_corners: list

    """
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())
    print('\nThe corner points are ...\n')
    for index, c in enumerate(approx_corners):
        character = chr(65 + index)
        print(character, ':', c)
        cv2.putText(canvas, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Rearranging the order of the corner points
    approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]

    plt.imshow(canvas)
    plt.title('Corner Points: Douglas-Peucker')
    plt.show()
    return approx_corners

def example_two(img_path):
    """
    Skew correction using homography and corner detection using contour points
    Returns: None

    """
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title('Original Image')
    plt.show()

    filtered_image = apply_filter(image)
    threshold_image = apply_threshold(filtered_image)

    cnv, largest_contour = detect_contour(threshold_image, image.shape)
    corners = detect_corners_from_contour(cnv, largest_contour)

    destination_points, h, w = get_destination_points(corners)
    un_warped = unwarp(image, np.float32(corners), destination_points)

    cropped = un_warped[0:h, 0:w]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    # f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(un_warped)
    ax2.imshow(cropped)

    plt.show()


if __name__ == '__main__':
    # example_one()
    img_path = r"D:\engine\cv\image_restoration\backup\18-48\plate_saved\2024-10-24-18-40-07-071078.jpg"
    # img_path = r"D:\engine\cv\image_restoration\backup\18-48\plate_saved\2024-10-24-18-32-55-343434.jpg"
    example_two(img_path=img_path)

# import numpy as np
# import cv2
# from skimage import io
# from skimage.transform import rotate
# from skimage.color import rgb2gray
# from deskew import determine_skew

# def deskew(_img):
#     image = io.imread(_img)
#     grayscale = rgb2gray(image)
#     angle = determine_skew(grayscale)
#     rotated = rotate(image, angle, resize=True) * 255
#     return rotated.astype(np.uint8)

# def display_before_after(_original):
#     # Load the original image using OpenCV
#     original_image = cv2.imread(_original)

#     # Deskew the image
#     deskewed_image = deskew(_original)

#     # Convert deskewed image from RGB to BGR
#     deskewed_image_bgr = cv2.cvtColor(deskewed_image, cv2.COLOR_RGB2BGR)

#     # Display original and deskewed images using OpenCV
#     cv2.imshow('Original Image', original_image)
#     cv2.imshow('Deskewed Image', deskewed_image_bgr)
    
#     # Wait for a key press and close the image windows
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Update the path to your image
# display_before_after(r'C:\Users\DOT\Documents\febri\github\ai-smartparking\plate_saved\2024-10-23-10-37-51-918009.jpg')


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