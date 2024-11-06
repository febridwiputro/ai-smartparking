import cv2
import numpy as np
import logging
import os
import random

def check_background(gray_image, verbose=False):
    white_threshold = 50
    _, white_mask = cv2.threshold(gray_image, white_threshold, 255, cv2.THRESH_BINARY)
    _, black_mask = cv2.threshold(gray_image, white_threshold, 255, cv2.THRESH_BINARY_INV)

    white_count = np.sum(white_mask == 255)
    black_count = np.sum(black_mask == 255)

    dominant_color = "bg_white" if white_count > black_count else "bg_black"
    print("white_count:", white_count, "black_count:", black_count)

    folder_path = "gray_images/white" if dominant_color == "bg_white" else "gray_images/black"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    filename = f"{folder_path}/{random.randint(1000, 9999)}.png"
    cv2.imwrite(filename, gray_image)

    if verbose:
        logging.info(f"Dominant background color detected: {dominant_color.upper()}")

    return dominant_color


def process_images_from_folder(input_folder):
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            gray_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if gray_image is None:
                print(f"Error: Gambar {filename} tidak ditemukan atau gagal dibaca.")
                continue

            dominant_color = check_background(gray_image, verbose=True)
            print(f"Warna latar belakang dominan untuk {filename}: {dominant_color}")


def process_single_image(image_path):
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if gray_image is None:
        print(f"Error: Gambar {image_path} tidak ditemukan atau gagal dibaca.")
        return
    
    dominant_color = check_background(gray_image, verbose=True)
    print(f"Warna latar belakang dominan yang terdeteksi: {dominant_color}")

def main():
    # input_folder = r"D:\engine\smart_parking\repository\github\ai-smartparking\gray_images"
    input_folder = r"D:\engine\smart_parking\repository\github\ai-smartparking\gray_image\selection"
    process_images_from_folder(input_folder)

if __name__ == "__main__":
    main()
