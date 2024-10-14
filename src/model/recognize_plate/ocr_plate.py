import logging, os
import cv2
import numpy as np
import tensorflow as tf
from os.path import splitext, basename
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import random
import glob
from datetime import datetime
from colorama import Fore, Style, init
from ultralytics import YOLO
import re

# Initialize colorama
init(autoreset=True)

import os, sys
import logging
this_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(this_path)

from config import Config
from show_ocr_plate import display_character_segments, display_results
from utils.backgrounds import check_background

# logging.disable(logging.WARNING)
# os.environ["TF_ENABLE_ONEDNN_OPTS"]= "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OCRPlate:
    def __init__(self, model_path, weight_path, labels_path, threshold=0.30):
        self.model = self.load_model(model_path, weight_path)
        self.labels = self.load_labels(labels_path)
        self.threshold = threshold
        self.yolo = YOLO(r"D:\engine\smart_parking\train_model\model-training\yolo\runs\classify\train3\weights\best.pt") 

    def load_model(self, model_path, weight_path):
        try:
            with open(model_path, 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json)
            model.load_weights(weight_path)
            print("[INFO] Model loaded successfully...")
            return model
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
            return None

    def load_labels(self, labels_path):
        try:
            labels = LabelEncoder()
            labels.classes_ = np.load(labels_path)
            print("[INFO] Labels loaded successfully...")
            return labels
        except Exception as e:
            print(f"[ERROR] Could not load labels: {e}")
            return None

    def preprocess_image(self, image_path, resize=False):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        if resize:
            img = cv2.resize(img, (224, 224))
        return img

    def predict_from_model(self, image):
        image = cv2.resize(image, (80, 80))
        image = np.stack((image,) * 3, axis=-1)

        predictions = self.model.predict(image[np.newaxis, :])
        max_prob = np.max(predictions)
        predicted_class = np.argmax(predictions)

        if max_prob >= self.threshold:
            prediction = self.labels.inverse_transform([predicted_class])
            return prediction[0], max_prob
        else:
            return None, max_prob

    def sort_contours(self, cnts, reverse=False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
        return cnts

    def segment_characters_black(self, binary_image, original_image, is_save, output_dir, is_inverse):

        cv2.imshow("binary", binary_image)

        cont, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        test_img1 = original_image.copy()
        img_res = []
        img_res_resized = []
        digit_w, digit_h = 30, 60

        for c in self.sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if 1 <= ratio <= 6:
                # logging.info(f'ALL: {h / original_image.shape[0]}')
                if 0.3 <= h / original_image.shape[0] <= 0.9: # 0.3 <= VALUE <= 0.5 
                    # logging.info(f'PASS: {h / original_image.shape[0]}')
                    cv2.rectangle(test_img1, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    curr_num = binary_image[y:y + h, x:x + w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    img_res.append(curr_num)

                    char_resized = cv2.resize(curr_num, (20, 30))
                    img_res_resized.append(char_resized)

                    if is_save:
                        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                        random_int = random.randint(100, 999)
                        filename = f"black-{timestamp}-{random_int}.png"
                        output_path = os.path.join(output_dir, filename)

                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        cv2.imwrite(output_path, char_resized)
                        print(f"[INFO] Saved {output_path}")

        return img_res_resized, test_img1, test_img1

    # def find_contours(self, dimensions, img, original_image, is_save, output_dir, is_inverse):
    #     if is_inverse:
    #         img_inv = cv2.bitwise_not(img)
    #     else:
    #         img_inv = img.copy()

    #     cntrs, _ = cv2.findContours(img_inv.copy(), 
    #                                 cv2.RETR_TREE, 
    #                                 cv2.CHAIN_APPROX_SIMPLE)

    #     sorted_cntrs = self.sort_contours(cntrs)

    #     lower_width, upper_width, lower_height, upper_height = dimensions
    #     ii = original_image.copy()
    #     x_cntr_list = []
    #     img_res = []
    #     img_res_resized = []

    #     for cntr in sorted_cntrs:
    #         intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
    #         ratio = intHeight / intWidth

    #         logging.info(f'WIDTH: {intWidth}, HEIGHT: {intHeight}')

    #         if (intWidth >= 10 and intWidth <= 50) and (intHeight > 30):
    #             logging.info('=' * 20 + " BORDER: find_contours " + '=' * 20)
    #             logging.info(f'PASS WIDTH: {intWidth}, HEIGHT: {intHeight}')
    #             logging.info(f'PASS: {intWidth}x{intHeight} with RATIO: {ratio}')

    #             x_cntr_list.append(intX)
    #             char = img_inv[intY:intY + intHeight, intX:intX + intWidth]
    #             char = cv2.bitwise_not(char)
    #             img_res.append(char)

    #             char_resized = cv2.resize(char, (20, 30))
    #             img_res_resized.append(char_resized)

    #             cv2.rectangle(ii, (intX, intY), (intX + intWidth, intY + intHeight), (255, 255, 0), 1)
    #             if is_save:
    #                 timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    #                 random_int = random.randint(100, 999)
    #                 filename = f"white-{timestamp}-{random_int}.png"
    #                 output_path = os.path.join(output_dir, filename)

    #                 if not os.path.exists(output_dir):
    #                     os.makedirs(output_dir)

    #                 cv2.imwrite(output_path, char_resized)
    #                 print(f"[INFO] Saved {output_path}")

    #     return img_res_resized, ii

    ################################################################################
    
    # def find_contours(self, dimensions, img, original_image, is_save, output_dir, is_inverse):
    #     if is_inverse:
    #         img_inv = cv2.bitwise_not(img)
    #     else:
    #         img_inv = img.copy()

    #     cntrs, _ = cv2.findContours(img_inv.copy(), 
    #                                 cv2.RETR_TREE, 
    #                                 cv2.CHAIN_APPROX_SIMPLE)

    #     sorted_cntrs = self.sort_contours(cntrs)

    #     lower_width, upper_width, lower_height, upper_height = dimensions
    #     ii = original_image.copy()
    #     x_cntr_list = []
    #     img_res = []
    #     img_res_resized = []
        
    #     # List to store the heights of valid contours
    #     valid_contour_heights = []

    #     # First loop to collect valid contour heights
    #     for cntr in sorted_cntrs:
    #         intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
    #         ratio = intHeight / intWidth

    #         logging.info(f'WIDTH: {intWidth}, HEIGHT: {intHeight}')

    #         # Check if contour dimensions are valid
    #         if (intWidth >= 10) and (intHeight > 30):
    #             valid_contour_heights.append(intHeight)
        
    #     # Calculate the average height of valid contours
    #     avg_height = sum(valid_contour_heights) / len(valid_contour_heights) if valid_contour_heights else 0
    #     logging.info(f'Average HEIGHT: {avg_height}')

    #     # Second loop to process contours and apply filtering based on height difference
    #     for cntr in sorted_cntrs:
    #         intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
    #         ratio = intHeight / intWidth

    #         # Only process contours that meet the size criteria
    #         if (intWidth >= 10 ) and (intHeight > 30):
    #             # Filter out contours with height difference >= 10 from the average height
    #             if abs(intHeight - avg_height) >= 10:
    #                 logging.info(f'Contour with HEIGHT: {intHeight} removed due to height difference')
    #                 continue

    #             logging.info(f'PASS WIDTH: {intWidth}, HEIGHT: {intHeight}')
    #             logging.info(f'PASS: {intWidth}x{intHeight} with RATIO: {ratio}')

    #             x_cntr_list.append(intX)
    #             char = img_inv[intY:intY + intHeight, intX:intX + intWidth]
    #             char = cv2.bitwise_not(char)
    #             img_res.append(char)

    #             char_resized = cv2.resize(char, (20, 30))
    #             img_res_resized.append(char_resized)

    #             cv2.rectangle(ii, (intX, intY), (intX + intWidth, intY + intHeight), (255, 255, 0), 1)
                
    #             if is_save:
    #                 timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    #                 random_int = random.randint(100, 999)
    #                 filename = f"white-{timestamp}-{random_int}.png"
    #                 output_path = os.path.join(output_dir, filename)

    #                 if not os.path.exists(output_dir):
    #                     os.makedirs(output_dir)

    #                 cv2.imwrite(output_path, char_resized)
    #                 print(f"[INFO] Saved {output_path}")

    #     return img_res_resized, ii

    #######################

    def find_contours(self, dimensions, img, original_image, is_save, output_dir, is_inverse):
        if is_inverse:
            img_inv = cv2.bitwise_not(img)
        else:
            img_inv = img.copy()

        cntrs, _ = cv2.findContours(img_inv.copy(), 
                                    cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE)

        sorted_cntrs = self.sort_contours(cntrs)

        lower_width, upper_width, lower_height, upper_height = dimensions
        ii = original_image.copy()
        x_cntr_list = []
        img_res = []
        img_res_resized = []
        
        # List to store the heights of valid contours
        valid_contour_heights = []

        # Collect heights of contours that meet the initial width and height criteria
        for cntr in sorted_cntrs:
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

            logging.info(f'WIDTH: {intWidth}, HEIGHT: {intHeight}')

            if (intWidth >= 10 and intWidth <= 50) and (intHeight > 30):
                valid_contour_heights.append(intHeight)

        # If there are no valid contour heights, return early
        if not valid_contour_heights:
            return img_res_resized, ii

        # Sort the valid heights in descending order to get the top two heights
        valid_contour_heights.sort(reverse=True)

        # Get the second highest height, or the highest if there's only one valid contour
        highest_height = valid_contour_heights[0]
        second_highest_height = valid_contour_heights[1] if len(valid_contour_heights) > 1 else highest_height

        logging.info(f'HIGHEST HEIGHT: {highest_height}, SECOND HIGHEST HEIGHT: {second_highest_height}')

        # Process the contours again, filtering based on the second highest height difference
        for cntr in sorted_cntrs:
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
            ratio = intHeight / intWidth

            # Check if contour meets the height difference criterion
            if (intWidth >= 10 and intWidth <= 50) and (intHeight > 30):
                logging.info('=' * 25 + f' BORDER: FIND CONTOURS CHARACTERS ' + '=' * 25)
                height_difference = abs(intHeight - second_highest_height)

                # Log the height difference
                logging.info(f'HEIGHT: {intHeight}, DIFFERENCE WITH SECOND HIGHEST: {height_difference}')

                # Filter based on the height difference
                if height_difference >= 10:
                    logging.info(f'Contour with HEIGHT: {intHeight} removed due to height difference')
                    continue

                logging.info(f'PASS WIDTH: {intWidth}, HEIGHT: {intHeight}')
                logging.info(f'PASS: {intWidth}x{intHeight} with RATIO: {ratio}')

                x_cntr_list.append(intX)
                char = img_inv[intY:intY + intHeight, intX:intX + intWidth]
                char = cv2.bitwise_not(char)
                img_res.append(char)

                char_resized = cv2.resize(char, (20, 30))
                img_res_resized.append(char_resized)

                # Draw rectangle around the contour
                cv2.rectangle(ii, (intX, intY), (intX + intWidth, intY + intHeight), (255, 255, 0), 1)
                
                # Save the image if required
                if is_save:
                    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    random_int = random.randint(100, 999)
                    filename = f"white-{timestamp}-{random_int}.png"
                    output_path = os.path.join(output_dir, filename)

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    cv2.imwrite(output_path, char_resized)
                    print(f"[INFO] Saved {output_path}")

        return img_res_resized, ii


    def segment_characters_white(self, image, is_save, output_dir, is_inverse):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (1, 1), 0)
        inv = cv2.bitwise_not(blur)
        _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        LP_WIDTH, LP_HEIGHT = binary.shape
        dimensions = [max(1, LP_WIDTH / 10), LP_WIDTH / 2, max(1, LP_HEIGHT / 10), 2 * LP_HEIGHT / 3]

        char_list, ii = self.find_contours(dimensions=dimensions, img=binary, original_image=image, is_save=is_save, output_dir=output_dir, is_inverse=is_inverse)
        inv = cv2.cvtColor(inv, cv2.COLOR_BGR2RGB)

        return char_list, ii, inv
    
    def match_char(self, plate):
        pattern = r"^(.{2})(.{0,4})(.*?)(.{2})$"

        def replace(match):
            prefix = match.group(1)
            middle = match.group(2)
            body = match.group(3)
            suffix = match.group(4)

            modified_middle = middle.replace('T', '1').replace('I', '1').replace('D', '0')
            modified_suffix = suffix.replace('0', 'Q')

            return f"{prefix}{modified_middle}{body}{modified_suffix}"

        result = re.sub(pattern, replace, plate)
        return result

    def process_image(self, img_path, is_inverse=True):
        if img_path is None:
            print(f"[ERROR] Failed to load image: {img_path}")
            return

        if type(img_path) == str:
            if not os.path.exists(img_path):
                print(f"[ERROR] File not found: {img_path}")
                return
            img = cv2.imread(img_path)

        else:
            img = img_path

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray, bg_status = check_background(gray)

        if bg_status == "bg_black":
            blur = cv2.GaussianBlur(gray, (3, 5), 0)
            binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

            crop_characters, segmented_image, inv_image = self.segment_characters_black(thre_mor, rgb, is_save=False, output_dir="output_chars_3", is_inverse=False)

            final_string = ''
            result_string = ''
            for i, character in enumerate(crop_characters):
                predicted_char, confidence = self.predict_from_model(character)
                if predicted_char:
                    final_string += predicted_char
                    result_string += f"Char: {predicted_char}, Conf: {confidence:.2f}\n"
                else:
                    result_string += f"Character below confidence threshold, Confidence: {confidence:.2f}\n"

            # display_results(rgb, inv_image, segmented_image, crop_characters, final_string, result_string)
            # logging.info(f'PLATE NO: {final_string}')
            logging.info('=' * 20 + f' PLATE NO: {final_string} ' + '=' * 20)
        else:
            char_list, img_segment, inv_image = self.segment_characters_white(rgb, is_save=False, output_dir="output_chars_3", is_inverse=True)

            final_string = ''
            result_string = ''  # String to store character and confidence info for display
            for i, character in enumerate(char_list):
                predicted_char, confidence = self.predict_from_model(character)
                if predicted_char:
                    final_string += predicted_char
                    result_string += f"Char: {predicted_char}, Conf: {confidence:.2f}\n"
                else:
                    result_string += f"Character below confidence threshold, Confidence: {confidence:.2f}\n"


            logging.info('=' * 20 + f' BEFORE PLATE NO: {final_string} ' + '=' * 20)
            final_string = self.match_char(final_string)
            logging.info('=' * 20 + f' AFTER PLATE NO: {final_string} ' + '=' * 20)

            display_results(rgb, inv_image, img_segment, char_list, final_string, result_string)

    def process_image_folder(self, folder_path):
        image_extensions = ["*.jpg", "*.png", "*.webp"]
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

        if not image_paths:
            logging.warning("No images (.jpg, .png, .webp) found in the folder.")
            return

        for img_path in image_paths:
            # logging.info(f"Processing image: {img_path}")
            self.process_image(img_path)

if __name__ == '__main__':
    model_path, weight_path, labels_path = Config.get_model_paths()
    
    ocr = OCRPlate(model_path, weight_path, labels_path)

    if Config.IS_FOLDER:
        if Config.IS_RESTORATION:
            folder_path = Config.BLACK_PLATE_RESTORATION_PATH if Config.IS_BLACK_PLATE else Config.WHITE_PLATE_RESTORATION_PATH
        else:
            folder_path = Config.BLACK_PLATE_PATH if Config.IS_BLACK_PLATE else Config.WHITE_PLATE_PATH

        ocr.process_image_folder(folder_path)
    else:
        if Config.IS_RESTORATION:
            img_path = Config.IMG_PATH_BLACK_RESTORATION if Config.IS_BLACK_PLATE else Config.IMG_PATH_WHITE_RESTORATION
        else:
            img_path = Config.IMG_PATH_BLACK if Config.IS_BLACK_PLATE else Config.IMG_PATH_WHITE

        ocr.process_image(img_path)