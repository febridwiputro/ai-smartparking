import os
import cv2
import numpy as np
import logging
from datetime import datetime

import os, sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import glob
import re

from src.models.display.character_recognition_display import display_character_segments, display_results

this_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(this_path)

from src.config.config import config
from src.config.logger import Logger

# Set TF_CPP_MIN_LOG_LEVEL to suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging = Logger("char_recog_model", is_save=False)
from src.config.config import config
from src.config.logger import logger
from src.models.gan_model import GanModel



def image_restoration(stopped, plate_result_queue, img_restore_text_char_queue):
    img_restore = ImageRestoration()
    text_detection = TextDetector()
    character_recognition = CharacterRecognize()

    while not stopped.is_set():
        # try:
        plate_result = plate_result_queue.get()

        if plate_result is None or len(plate_result) == 0:
            continue

        plate_image = plate_result.get("frame", None)
        bg_color = plate_result.get("bg_color", None)
        floor_id = plate_result.get("floor_id", 0)
        cam_id = plate_result.get("cam_id", "")
        arduino_idx = plate_result.get("arduino_idx", None)
        car_direction = plate_result.get("car_direction", None)
        start_line = plate_result.get("start_line", None)
        end_line = plate_result.get("end_line", None)

        if plate_image is None:
            continue

        empty_frame = np.empty((0, 0, 3), dtype=np.uint8)

        if not start_line and not end_line:
            result = {
                "frame": empty_frame,
                "floor_id": floor_id,
                "cam_id": cam_id,
                "arduino_idx": arduino_idx,
                "car_direction": car_direction,
                "start_line": start_line,
                "end_line": end_line
            }
            img_restore_text_char_queue.put(result)
            continue

        if plate_image is None:
            continue

        restored_image = img_restore.process_image(plate_image)
        text_detected_result, _ = text_detection.easyocr_readtext(image=restored_image)
        plate_no = character_recognition.process_image(text_detected_result, bg_color) if text_detected_result is not None else ""

        logging.write(f'PLATE_NO: {plate_no}', logging.DEBUG)

        result = {
            "plate_no": plate_no,
            "floor_id": floor_id,
            "cam_id": cam_id,
            "arduino_idx": arduino_idx,
            "car_direction": car_direction,
            "start_line": start_line,
            "end_line": end_line
        }

        img_restore_text_char_queue.put(result)

        # except Exception as e:
        #     print(f"Error in image_restoration: {e}")

class ImageRestoration:
    def __init__(self):
        self.gan_model = GanModel()
        self.saved_dir = 'image_restoration_saved'

    def process_image(self, image, is_save=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

        restored_image = self.gan_model.super_resolution(resized_image)

        if is_save:
            self.save_restored_image(restored_image)

        return restored_image

    def save_restored_image(self, restored_image):
        if not os.path.exists(self.saved_dir):
            os.makedirs(self.saved_dir)

        if restored_image is not None and restored_image.size > 0:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            filename = os.path.join(self.saved_dir, f'{timestamp}.jpg')

            if len(restored_image.shape) == 2:  # Grayscale
                cv2.imwrite(filename, restored_image)
            elif len(restored_image.shape) == 3:  # Color
                cv2.imwrite(filename, cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR))

            # logging.info(f"[ImageRestoration] Image saved as {filename}")
        else:
            logging.warning("[ImageRestoration] Restored image is empty or invalid, not saving.")


import os
import cv2
import numpy as np
from easyocr import Reader
import time
from datetime import datetime

from src.config.config import config
from src.config.logger import Logger
from src.models.utils.text_detection_util import (
    filter_height_bbox, 
    filter_readtext_frame, 
    save_cropped_images
)

logging = Logger("text_detection_model", is_save=False)


class EasyOCRNet:
    def __init__(self, languages=['en'], use_cuda=True):
        """
        Initialize EasyOCR Reader
        """
        self.reader = Reader(languages, gpu=use_cuda, verbose=False)

    def detect(self, image):
        """
        Detect text in the image.
        Returns bounding boxes.
        """
        t0 = time.time()
        results = self.reader.detect(image)
        t1 = time.time() - t0
        return results[0] if results else []

    def readtext(self, image):
        """
        Read text from the image.
        Returns text and bounding boxes.
        """
        t0 = time.time()
        results = self.reader.readtext(image,
                                       text_threshold=0.7,
                                       low_text=0.4,
                                       decoder='greedy',
                                       slope_ths=0.6,
                                       add_margin=0.0
                                       )
        t1 = time.time() - t0
        if results:
            return results
        else:
            logging.write(f"No text recognized.", logging.DEBUG)
            return []

class TextDetector:
    def __init__(self):
        self.ocr_net = EasyOCRNet(use_cuda=True)

    def easyocr_detect(self, image, is_save=False):
        bounding_boxes = self.ocr_net.detect(image)
        filtered_heights = filter_height_bbox(bounding_boxes=bounding_boxes)

        converted_bboxes = []
        cropped_images = []
        
        for bbox_group in bounding_boxes:
            for bbox in bbox_group:
                if len(bbox) == 4:
                    x_min, x_max, y_min, y_max = bbox
                    top_left = [x_min, y_min]
                    bottom_right = [x_max, y_max]

                    width_bbox = x_max - x_min
                    height_bbox = y_max - y_min

                    if height_bbox not in filtered_heights:
                        continue

                    top_left_y, bottom_right_y = int(max(top_left[1], 0)), int(min(bottom_right[1], image.shape[0]))
                    top_left_x, bottom_right_x = int(max(top_left[0], 0)), int(min(bottom_right[0], image.shape[1]))

                    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

                    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                        logging.write(f"Skipped empty cropped image with shape: {cropped_image.shape}", logging.DEBUG)
                        continue

                    cropped_images.append(cropped_image)

        if cropped_images:
            if is_save:
                save_cropped_images(cropped_images)

        # Return cropped images and processed frame
        return cropped_images, image

    def easyocr_readtext(self, image):

        bounding_boxes = self.ocr_net.readtext(image)

        cropped_images = []
        filtered_heights = filter_readtext_frame(bounding_boxes, False)

        for t in bounding_boxes:
            (top_left, top_right, bottom_right, bottom_left) = t[0]
            top_left = tuple([int(val) for val in top_left])
            bottom_left = tuple([int(val) for val in bottom_left])
            top_right = tuple([int(val) for val in top_right])

            height_f = bottom_left[1] - top_left[1]
            width_f = top_right[0] - top_left[0]

            # logging.write('=' * 25 + f' BORDER: EASYOCR READTEXT' + '=' * 25, logging.DEBUG)
            # logging.write(f'height: {height_f}, width: {width_f}', logging.DEBUG)

            if height_f not in filtered_heights:
                continue

            top_left_y, bottom_right_y = int(max(top_left[1], 0)), int(min(bottom_right[1], image.shape[0]))
            top_left_x, bottom_right_x = int(max(top_left[0], 0)), int(min(bottom_right[0], image.shape[1]))

            cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                logging.write(f"Skipped empty cropped image with shape: {cropped_image.shape}", logging.DEBUG)
                continue

            cropped_images.append(cropped_image)

        return cropped_images, image

    def filter_height_bbox(self, bounding_boxes):
        heights = [box[3] - box[2] for group in bounding_boxes for box in group if len(box) == 4]
        return heights


class ModelAndLabelLoader:
    @staticmethod
    def load_model(model_path, weight_path):
        """Loads a Keras model from a JSON file and weights from an H5 file."""
        try:
            with open(model_path, 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json)
            model.load_weights(weight_path)
            logging.write(f'Model char_recognize loaded', logging.DEBUG)
            # logging.write(f'Model loaded from {model_path} and weights from {weight_path}', logging.DEBUG)
            return model
        except Exception as e:
            logging.write(f'Could not load model: {e}', logging.DEBUG)
            return None

    @staticmethod
    def load_labels(labels_path):
        """Loads labels using LabelEncoder from a NumPy file."""
        try:
            labels = LabelEncoder()
            labels.classes_ = np.load(labels_path)
            logging.write(f'Label char_recognize loaded', logging.DEBUG)
            # logging.write(f'Labels loaded from {labels_path}', logging.DEBUG)
            return labels
        except Exception as e:
            logging.write(f'Could not load labels: {e}', logging.DEBUG)
            return None


class CharacterRecognize:
    def __init__(self, threshold=0.30, models=None, labels=None):
        self.model = models
        self.labels = labels
        # model_path, weight_path, labels_path = config.MODEL_CHAR_RECOGNITION_PATH, config.WEIGHT_CHAR_RECOGNITION_PATH, config.LABEL_CHAR_RECOGNITION_PATH
        # self.model = self.load_model(model_path, weight_path)
        # self.labels = self.load_labels(labels_path)
        self.threshold = threshold

    def load_model(self, model_path, weight_path):
        try:
            with open(model_path, 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json)
            model.load_weights(weight_path)
            # logging.write(f'{os.path.basename(model_path)} & {os.path.basename(weight_path)} loaded successfully...', level=logging.INFO)

            return model

        except Exception as e:
            logging.write(f'Could not load model: {e}', logging.ERROR)

            return None

    def load_labels(self, labels_path):
        try:
            labels = LabelEncoder()
            labels.classes_ = np.load(labels_path)
            # logging.write(f'{os.path.basename(labels_path)} loaded successfully...', level=logging.INFO)

            return labels

        except Exception as e:
            logging.write(f'Could not load labels: {e}', logging.ERROR)

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

        predictions = self.model.predict(image[np.newaxis, :], verbose=False)
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

    def segment_characters_black(self, img_bgr, is_save, output_dir, verbose=False):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_copy = img_rgb.copy()

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (1, 1), 0)
        binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cont, _ = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_res = []
        img_res_resized = []
        digit_w, digit_h = 30, 60
        sorted_cntrs = self.sort_contours(cont)
        valid_contour_heights = []

        for cntr in sorted_cntrs:
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
            valid_contour_heights.append(intHeight)

        if not valid_contour_heights:
            if verbose:
                logging.write("No valid contours found", logging.DEBUG)
            return img_res_resized, img_rgb_copy, img_rgb_copy

        if verbose:
            logging.write(f'LIST: {valid_contour_heights}', logging.DEBUG)

        valid_contour_heights.sort(reverse=True)
        highest_height = valid_contour_heights[0]

        if verbose:
            logging.write(f'SORT: {valid_contour_heights}, TOTAL: {len(valid_contour_heights)}, HIGHEST: {highest_height}', logging.DEBUG)

        second_highest_height = valid_contour_heights[1] if len(valid_contour_heights) > 1 else highest_height
        if verbose:
            logging.write(f'HIGHEST HEIGHT: {highest_height}, SECOND HIGHEST HEIGHT: {second_highest_height}', logging.DEBUG)

        for cntr in sorted_cntrs:
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
            ratio = intHeight / intWidth

            if verbose:
                logging.write('=' * 25 + f' BORDER: SEGMENTATION CHARACTERS' + '=' * 25, logging.DEBUG)
                logging.write(f'WIDTH={intWidth} & HEIGHT={intHeight}', logging.DEBUG)

            height_difference = abs(second_highest_height - intHeight)

            if height_difference >= 20:
                if verbose:
                    logging.write(f'Contour with HEIGHT: {intHeight} removed due to height difference', logging.DEBUG)
                continue
            else:
                if intWidth >= intHeight:
                    if verbose:
                        logging.write(f'Contour with HEIGHT: {intHeight} removed due to invalid width-height ratio', logging.DEBUG)
                    continue
                if intHeight > 25 and intWidth < 5:
                    if verbose:
                        logging.write(f'Contour with HEIGHT: {intHeight} removed due to small width', logging.DEBUG)
                    continue
                elif intHeight <= 25 and intWidth <= 5:
                    if verbose:
                        logging.write(f'Contour with HEIGHT: {intHeight} removed due to small width', logging.DEBUG)
                    continue

                if intWidth >= 50:
                    if verbose:
                        logging.write(f'Contour with WIDTH: {intWidth} removed due to excessive width', logging.DEBUG)
                    continue

                if verbose:
                    logging.write(f'>>>>> RESULT: {height_difference} = {second_highest_height} - {intHeight}', logging.DEBUG)

            # Scale original image and convert to binary
            char = img_rgb_copy[intY:intY + intHeight, intX:intX + intWidth]
            char_gray = cv2.cvtColor(char, cv2.COLOR_RGB2GRAY)
            _, char_binary = cv2.threshold(char_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_res.append(char_binary)

            char_resized = cv2.resize(char, (20, 30))
            img_res_resized.append(char_resized)
            cv2.rectangle(img_rgb_copy, (intX, intY), (intX + intWidth, intY + intHeight), (0, 255, 0), 1)

            if is_save:
                timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                filename = f"black-{timestamp}.png"
                output_path = os.path.join(output_dir, filename)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                cv2.imwrite(output_path, char_resized)
                logging.write(f'Saved {output_path}', logging.DEBUG)

        return img_res, img_rgb_copy, img_rgb_copy

    def match_char(self, plate):
        pattern = r"^(.{2})(.{0,4})(.*?)(.{2})$"

        def replace(match):
            prefix = match.group(1)
            middle = match.group(2)
            body = match.group(3)
            suffix = match.group(4)

            modified_middle = middle.replace('T', '1').replace('I', '1').replace('D', '0').replace('B', '8').replace('Q', '0')

            if re.match(r"^[A-Z]{2}\d{4}$", f"{prefix}{modified_middle}", logging.DEBUG):
                modified_suffix = suffix.replace('0', 'Q').replace('8', 'O')
            else:
                modified_suffix = suffix

            modified_plate = f"{prefix}{modified_middle}{body}{modified_suffix}"
            match_special_case = re.match(r"(\d{4})(.*)(BP)$", modified_plate)
            if match_special_case:
                return f"BP{match_special_case.group(1)}{match_special_case.group(2)}"

            return modified_plate

        result = re.sub(pattern, replace, plate)
        return result

    def process_image(self, cropped_images, bg_status):
        bg_color = ""
        final_plate = ""
        resized_images = []

        min_height = min(img.shape[0] for img in cropped_images if img.shape[0] > 0)

        for img in cropped_images:
            original_height = img.shape[0]
            original_width = img.shape[1]

            if original_height > 0:
                new_width = int(original_width * (min_height / original_height))
                if new_width > 0:
                    resized_img = cv2.resize(img, (new_width, min_height))
                    resized_images.append(resized_img)
                else:
                    logging.warning(f'Skipped resizing due to invalid width: {new_width}', logging.DEBUG)
            else:
                logging.warning('Skipped resizing due to invalid image height', logging.DEBUG)

        if resized_images:
            channels = resized_images[0].shape[2] if len(resized_images[0].shape) == 3 else 1
            concatenated_image = resized_images[0]

            if bg_status == "bg_black":
                bg_color = "bg_black"
                color_separator = np.zeros((min_height, 10, channels), dtype=np.uint8)

            elif bg_status == "bg_white":
                bg_color = "bg_white"
                if channels == 3:
                    color_separator = np.ones((min_height, 10, channels), dtype=np.uint8) * 255
                else:
                    color_separator = np.ones((min_height, 10), dtype=np.uint8) * 255

            else:
                bg_color = "bg_red"
                if channels == 3:
                    color_separator = np.zeros((min_height, 10, channels), dtype=np.uint8)
                    color_separator[:, :, 2] = 255
                else:
                    color_separator = np.zeros((min_height, 10), dtype=np.uint8)

            for img in resized_images[1:]:
                if img.shape[0] != min_height:
                    logging.warning(f"Image height mismatch: Resizing image from {img.shape[0]} to {min_height}", logging.DEBUG)
                    img = cv2.resize(img, (img.shape[1], min_height))

                concatenated_image = cv2.hconcat([concatenated_image, color_separator, img])

            final_plate = self.process_character(concatenated_image, bg_color)

        else:
            logging.write("No valid images to merge", logging.DEBUG)

        return final_plate

    def process_character(self, img_bgr, bg_color, verbose=False):
        final_string = ''
        result_string = ''

        if bg_color == "bg_black":
            crop_characters, segmented_image, inv_image = self.segment_characters_black(img_bgr, is_save=False, output_dir="output_chars_3", verbose=False)

            for i, character in enumerate(crop_characters):
                predicted_char, confidence = self.predict_from_model(character)
                if predicted_char:
                    final_string += predicted_char
                    result_string += f"Char: {predicted_char}, Conf: {confidence:.2f}\n"
                else:
                    result_string += f"Character below confidence threshold, Confidence: {confidence:.2f}\n"

            if verbose:
                logging.write('=' * 20 + f' BEFORE PLATE NO: {final_string} ' + '=' * 20, logging.DEBUG)
            final_string = self.match_char(final_string)
            if verbose:
                logging.write('=' * 20 + f' AFTER PLATE NO: {final_string} ' + '=' * 20, logging.DEBUG)

            display_results(img_bgr, inv_image, segmented_image, crop_characters, final_string, result_string, is_save=True)

            return final_string

        elif bg_color == "bg_white":
            img_inv = cv2.bitwise_not(img_bgr)
            char_list, img_segment, inv_image = self.segment_characters_black(img_inv, is_save=False, output_dir="output_chars_3", verbose=False)

            for i, character in enumerate(char_list):
                predicted_char, confidence = self.predict_from_model(character)
                if predicted_char:
                    final_string += predicted_char
                    result_string += f"Char: {predicted_char}, Conf: {confidence:.2f}\n"
                else:
                    result_string += f"Character below confidence threshold, Confidence: {confidence:.2f}\n"

            if verbose:
                logging.write('=' * 20 + f' BEFORE PLATE NO: {final_string} ' + '=' * 20, logging.DEBUG)
            final_string = self.match_char(final_string)
            if verbose:
                logging.write('=' * 20 + f' AFTER PLATE NO: {final_string} ' + '=' * 20, logging.DEBUG)

            display_results(img_bgr, inv_image, img_segment, char_list, final_string, result_string, is_save=True)

            return final_string

    def process_image_folder(self, folder_path):
        image_extensions = ["*.jpg", "*.png", "*.webp"]
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

        if not image_paths:
            logging.warning("No images (.jpg, .png, .webp) found in the folder.", logging.DEBUG)
            return

        for img_path in image_paths:
            logging.write(f"Processing image: {img_path}", logging.DEBUG)
            self.process_image(img_path)