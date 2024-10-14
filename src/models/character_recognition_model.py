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

def character_recognition(stopped, text_detection_result_queue, char_recognize_result_queue):
    char_model_path = config.MODEL_CHAR_RECOGNITION_PATH
    char_weight_path = config.WEIGHT_CHAR_RECOGNITION_PATH
    label_path = config.LABEL_CHAR_RECOGNITION_PATH

    model = ModelAndLabelLoader.load_model(char_model_path, char_weight_path)
    labels = ModelAndLabelLoader.load_labels(label_path)

    cr = CharacterRecognize(models=model, labels=labels)

    while not stopped.is_set():
        try:
            result = text_detection_result_queue.get()

            if result is None:
                continue

            bg_color = result.get("bg_color", None)
            cropped_images = result.get("frame", None)
            floor_id = result.get("floor_id", 0)
            cam_id = result.get("cam_id", "")
            arduino_idx = result.get("arduino_idx", None)
            car_direction = result.get("car_direction", None)
            mobil_masuk = result.get("mobil_masuk", None)
            passed = result.get("passed", 0)
            start_line = result.get("start_line", False)  # Default to False
            end_line = result.get("end_line", False)  # Default to False

            # print(f'start_line: {start_line} & end_line: {end_line}')

            empty_frame = np.empty((0, 0, 3), dtype=np.uint8)

            if not start_line and not end_line:
                char_recognize_result = {
                    "bg_color": bg_color,
                    "plate_no": "",  # or None, based on your requirement
                    "floor_id": floor_id,
                    "cam_id": cam_id,
                    "arduino_idx": arduino_idx,
                    "car_direction": car_direction,
                    "mobil_masuk": mobil_masuk,
                    "passed": passed,
                    "start_line": start_line,
                    "end_line": end_line
                }
                char_recognize_result_queue.put(char_recognize_result)
                continue

            plate_no = cr.process_image(cropped_images, bg_color) if cropped_images is not None else ""
            # logging.write(f'PLATE_NO: {plate_no}', logging.DEBUG)
            char_recognize_result = {
                "bg_color": bg_color,
                "plate_no": plate_no,
                "floor_id": floor_id,
                "cam_id": cam_id,
                "arduino_idx": arduino_idx,
                "car_direction": car_direction,
                "mobil_masuk": mobil_masuk,
                "passed": passed,
                "start_line": start_line,
                "end_line": end_line
            }

            char_recognize_result_queue.put(char_recognize_result)

        except Exception as e:
            print(f"Error in character recognition: {e}")


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

            display_results(img_bgr, inv_image, segmented_image, crop_characters, final_string, result_string, is_save=False)

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

            display_results(img_bgr, inv_image, img_segment, char_list, final_string, result_string, is_save=False)

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