import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# this_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# sys.path.append(this_path)

# from src.controllers.utils.util import (
#     check_background
# )

def shi_tomashi(image, is_show=False):
    """
    Use Shi-Tomashi algorithm to detect corners

    Args:
        image: np.array

    Returns:
        corners: list
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 100)
    if corners is None:
        print("No corners found.")
        return []
    
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

    if is_show:
        plt.imshow(im)
        plt.title('Corner Detection: Shi-Tomashi')
        plt.show()

    return corners

def get_destination_points(corners):
    """
    Get destination points from corners for homography transform

    Args:
        corners: list

    Returns:
        destination_corners: list
        height: int
        width: int
    """
    if len(corners) < 4:
        print("Insufficient corners for perspective transform.")
        return None, None, None

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
    Unwarp the image based on the source and destination points

    Args:
        img: np.array
        src: list
        dst: list

    Returns:
        un_warped: np.array
    """
    h, w = img.shape[:2]
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        print("Failed to calculate homography matrix.")
        return img  # Return original if homography fails
    print('\nThe homography matrix is: \n', H)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
    return un_warped

def detect_contour(img, image_shape, is_show=False):
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

    if is_show:
        plt.title('Largest Contour')
        plt.imshow(canvas)
        plt.show()

    return canvas, cnt

def apply_filter(image, is_show=False):
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
    if is_show:
        plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        plt.title('Filtered Image')
        plt.show()
    return filtered

def apply_threshold(filtered, is_show=False):
    """
    Apply OTSU threshold

    Args:
        filtered: np.array

    Returns:
        thresh: np.array

    """
    ret, thresh = cv2.threshold(filtered, 250, 255, cv2.THRESH_OTSU)
    
    if is_show:
        plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
        plt.title('After applying OTSU threshold')
        plt.show()
    return thresh

def show_bounding_boxes(thresh, verbose=True):
    """
    Display bounding boxes around detected characters on the thresholded image with contour filtering

    Args:
        thresh: np.array (binary image from thresholding)
        verbose: bool (if True, enable detailed logging)

    Returns:
        None (only displays the image with bounding boxes)
    """
    img_with_boxes = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by height and apply filtering conditions
    valid_contour_heights = [cv2.boundingRect(cntr)[3] for cntr in contours]
    valid_contour_heights.sort(reverse=True)
    highest_height = valid_contour_heights[0] if valid_contour_heights else 0
    second_highest_height = valid_contour_heights[1] if len(valid_contour_heights) > 1 else highest_height

    if verbose:
        print(f'SORT: {valid_contour_heights}, TOTAL: {len(valid_contour_heights)}, HIGHEST: {highest_height}')
        print(f'HIGHEST HEIGHT: {highest_height}, SECOND HIGHEST HEIGHT: {second_highest_height}')

    for cntr in contours:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        ratio = intHeight / intWidth

        # Calculate height difference and apply filters
        height_difference = abs(second_highest_height - intHeight)

        if height_difference >= 20:
            if verbose:
                print(f'Contour with HEIGHT: {intHeight} removed due to height difference')
            continue

        if intWidth >= intHeight:
            if verbose:
                print(f'Contour with HEIGHT: {intHeight} removed due to invalid width-height ratio')
            continue

        if intHeight > 25 and intWidth < 5:
            if verbose:
                print(f'Contour with HEIGHT: {intHeight} removed due to small width')
            continue
        elif intHeight <= 25 and intWidth <= 5:
            if verbose:
                print(f'Contour with HEIGHT: {intHeight} removed due to small width')
            continue

        if intWidth >= 50:
            if verbose:
                print(f'Contour with WIDTH: {intWidth} removed due to excessive width')
            continue

        if verbose:
            print(f'>>>>> RESULT: {height_difference} = {second_highest_height} - {intHeight}')

        # Draw a green bounding box around the filtered contour
        cv2.rectangle(img_with_boxes, (intX, intY), (intX + intWidth, intY + intHeight), (0, 255, 0), 1)

    # Display the image with bounding boxes
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title('Filtered Bounding Boxes on Thresholded Image')
    plt.show()

def detect_corners_from_contour(canvas, cnt, is_show=False):
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

    if is_show:
        plt.imshow(canvas)
        plt.title('Corner Points: Douglas-Peucker')
        plt.show()
    return approx_corners

def example_one(img_path):
    """
    Skew correction using homography and corner detection using Shi-Tomashi corner detector

    Returns: None

    """
    image = cv2.imread(img_path)
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

def example_two(img_path, is_show=False):
    """
    Skew correction using homography and corner detection using contour points
    """
    image = cv2.imread(img_path)
    if image is None:
        print("Image not found.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # bg_color = check_background(image, verbose=False, is_save=True)
    # print(bg_color)

    if is_show:
        plt.imshow(image)
        plt.title('Original Image')
        plt.show()

    # Apply filter and threshold
    filtered_image = apply_filter(image, is_show=False)
    threshold_image = apply_threshold(filtered_image)
    show_bounding_boxes(threshold_image)

    # Detect contour and corners
    cnv, largest_contour = detect_contour(threshold_image, image.shape)
    corners = detect_corners_from_contour(cnv, largest_contour)

    # Check if corners are sufficient for transform
    if len(corners) < 4:
        print("Not enough corners detected for transformation.")
        return

    destination_points, h, w = get_destination_points(corners)
    if destination_points is None:
        print("Unable to define destination points.")
        return

    # Perform deskew with homography
    un_warped = unwarp(image, np.float32(corners), destination_points)
    cropped = un_warped[0:h, 0:w]

    if is_show:
        # Display results
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        ax1.imshow(un_warped)
        ax1.set_title('Unwarped Image')
        ax2.imshow(cropped)
        ax2.set_title('Cropped Plate Image')
        plt.show()

if __name__ == '__main__':
    img_path = r"D:\engine\smart_parking\repository\github\ai-smartparking\image_restoration_saved\2024-11-12-10-56-02-687488.jpg"
    # example_one(img_path=img_path)
    example_two(img_path=img_path)