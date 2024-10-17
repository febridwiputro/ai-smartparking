import cv2
import numpy as np
import logging
from colorama import Fore, Style, init

init(autoreset=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_background(gray_image, verbose=False):
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    white_pixels = np.sum(binary_image == 255)
    black_pixels = np.sum(binary_image == 0)

    bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

    green_pixels = np.sum(mask_green == 255)
    yellow_pixels = np.sum(mask_yellow == 255)
    red_pixels = np.sum(mask_red == 255)
    blue_pixels = np.sum(mask_blue == 255)

    if abs(black_pixels - blue_pixels) / max(black_pixels, blue_pixels) < 0.1:
        if verbose:
            logging.info('=' * 80)
            logging.info('=' * 30 + " BLACK-BLUE PLATE " + '=' * 30)
            logging.info('=' * 80)
        return gray_image, "bg_black_blue"

    if white_pixels > black_pixels and white_pixels > green_pixels and white_pixels > yellow_pixels and white_pixels > red_pixels and white_pixels > blue_pixels:
        if verbose:
            logging.info('=' * 80)
            logging.info('=' * 30 + " WHITE PLATE " + '=' * 30)
            logging.info('=' * 80)
        return gray_image, "bg_white"

    elif black_pixels > white_pixels and black_pixels > green_pixels and black_pixels > yellow_pixels and black_pixels > red_pixels and black_pixels > blue_pixels:
        if verbose:
            logging.info('=' * 80)
            logging.info('=' * 30 + " BLACK PLATE " + '=' * 30)
            logging.info('=' * 80)
        return gray_image, "bg_black"

    elif green_pixels > white_pixels and green_pixels > yellow_pixels and green_pixels > red_pixels and green_pixels > blue_pixels:
        return gray_image, "bg_green"

    elif yellow_pixels > white_pixels and yellow_pixels > green_pixels and yellow_pixels > red_pixels and yellow_pixels > blue_pixels:
        return gray_image, "bg_yellow"

    elif red_pixels > white_pixels and red_pixels > green_pixels and red_pixels > yellow_pixels and red_pixels > blue_pixels:
        return gray_image, "bg_red"

    elif blue_pixels > white_pixels and blue_pixels > green_pixels and blue_pixels > yellow_pixels and blue_pixels > red_pixels:
        return gray_image, "bg_blue"

    else:
        return gray_image, "bg_unknown"