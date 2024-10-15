import logging, os, sys
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import glob
from datetime import datetime
import re

from src.model.recognize_plate.utils.display import display_character_segments, display_results
from src.model.recognize_plate.utils.backgrounds import check_background

this_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(this_path)

from src.config.config import config
from src.config.logger import logger

# os.environ["TF_ENABLE_ONEDNN_OPTS"]= "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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
            # logger.write(f'{os.path.basename(model_path)} & {os.path.basename(weight_path)} loaded successfully...', level=logger.INFO)

            return model

        except Exception as e:
            logger.write(f'Could not load model: {e}', logger.ERROR)

            return None

    def load_labels(self, labels_path):
        try:
            labels = LabelEncoder()
            labels.classes_ = np.load(labels_path)
            # logger.write(f'{os.path.basename(labels_path)} loaded successfully...', level=logger.INFO)

            return labels

        except Exception as e:
            logger.write(f'Could not load labels: {e}', logger.ERROR)

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
                logging.info("No valid contours found")
            return img_res_resized, img_rgb_copy, img_rgb_copy

        if verbose:
            logging.info(f'LIST: {valid_contour_heights}')

        valid_contour_heights.sort(reverse=True)
        highest_height = valid_contour_heights[0]

        if verbose:
            logging.info(f'SORT: {valid_contour_heights}, TOTAL: {len(valid_contour_heights)}, HIGHEST: {highest_height}')

        second_highest_height = valid_contour_heights[0] if len(valid_contour_heights) > 1 else highest_height
        if verbose:
            logging.info(f'HIGHEST HEIGHT: {highest_height}, SECOND HIGHEST HEIGHT: {second_highest_height}')

        for cntr in sorted_cntrs:
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
            ratio = intHeight / intWidth

            if verbose:
                logging.info('=' * 25 + f' BORDER: RECOGNIZE ' + '=' * 25)
                logging.info(f'WIDTH={intWidth} & HEIGHT={intHeight}')
            height_difference = abs(second_highest_height - intHeight)

            if height_difference >= 20:
                if verbose:
                    logging.info(f'Contour with HEIGHT: {intHeight} removed due to height difference')
                continue
            else:
                if intWidth >= intHeight:
                    if verbose:
                        logging.info(f'Contour with HEIGHT: {intHeight} removed due to invalid width-height ratio')
                    continue
                if intHeight > 25 and intWidth < 5:
                    if verbose:
                        logging.info(f'Contour with HEIGHT: {intHeight} removed due to small width')
                    continue
                elif intHeight <= 25 and intWidth <= 5:
                    if verbose:
                        logging.info(f'Contour with HEIGHT: {intHeight} removed due to small width')
                    continue

                if intWidth >= 50:
                    if verbose:
                        logging.info(f'Contour with WIDTH: {intWidth} removed due to excessive width')
                    continue

                if verbose:
                    logging.info(f'>>>>> RESULT: {height_difference} = {second_highest_height} - {intHeight}')

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
                logging.info(f'Saved {output_path}')

        return img_res, img_rgb_copy, img_rgb_copy

    def match_char(self, plate):
        pattern = r"^(.{2})(.{0,4})(.*?)(.{2})$"

        def replace(match):
            prefix = match.group(1)
            middle = match.group(2)
            body = match.group(3)
            suffix = match.group(4)

            modified_middle = middle.replace('T', '1').replace('I', '1').replace('D', '0').replace('B', '8').replace('Q', '0')

            if re.match(r"^[A-Z]{2}\d{4}$", f"{prefix}{modified_middle}"):
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
    
    def filter_height_bbox(self, bounding_boxes, verbose=False):
        converted_bboxes = []
        w_h = []
        for bbox_group in bounding_boxes:
            for bbox in bbox_group:
                if len(bbox) == 4:
                    x_min, x_max, y_min, y_max = bbox
                    top_left = [x_min, y_min]
                    top_right = [x_max, y_min]
                    bottom_right = [x_max, y_max]
                    bottom_left = [x_min, y_max]
                    converted_bboxes.append([top_left, top_right, bottom_right, bottom_left])

                    width_bbox = x_max - x_min
                    height_bbox = y_max - y_min 

                    if height_bbox >= 10:
                        w_h.append(height_bbox)

        if len(w_h) == 1:
            list_of_height = w_h
            filtered_heights = w_h
            sorted_heights = w_h

        elif len(w_h) == 2:
            list_of_height = w_h
            filtered_heights = [max(w_h)]
            sorted_heights = sorted(w_h, reverse=True)

        elif len(w_h) > 2:
            list_of_height = w_h
            sorted_heights = sorted(w_h, reverse=True)
            # sorted_heights = sorted([h for w, h in w_h], reverse=True)
            highest_height_f = sorted_heights[0]
            avg_height = sum(sorted_heights) / len(sorted_heights)

            filtered_heights = [highest_height_f]
            filtered_heights += [h for h in sorted_heights[1:] if abs(highest_height_f - h) < 20]

        else:
            filtered_heights = w_h

        if verbose:
            logging.info('>' * 25 + f' BORDER: filter_height_bbox ' + '>' * 25)
            logging.info(f'LIST OF HEIGHT: {list_of_height}, SORTED HEIGHT: {sorted_heights}, FILTERED HEIGHTS: {filtered_heights}, AVG HEIGHT: {avg_height}')

        return filtered_heights      

    def filter_text_frame(self, texts: list, verbose=False) -> str:
        w_h = []
        sorted_heights = []
        avg_height = ""
        for t in texts:
            (top_left, top_right, bottom_right, bottom_left) = t[0]
            top_left = tuple([int(val) for val in top_left])
            bottom_left = tuple([int(val) for val in bottom_left])
            top_right = tuple([int(val) for val in top_right])
            height_f = bottom_left[1] - top_left[1]
            width_f = top_right[0] - top_left[0]

            if height_f >= 10:
                # w_h.append([width_f, height_f])
                w_h.append(height_f)

        if len(w_h) == 1:
            list_of_height = w_h
            filtered_heights = w_h
            sorted_heights = w_h

        elif len(w_h) == 2:
            list_of_height = w_h
            filtered_heights = [max(w_h)]
            sorted_heights = sorted(w_h, reverse=True)

        elif len(w_h) > 2:
            list_of_height = w_h
            sorted_heights = sorted(w_h, reverse=True)
            # sorted_heights = sorted([h for w, h in w_h], reverse=True)
            highest_height_f = sorted_heights[0]
            avg_height = sum(sorted_heights) / len(sorted_heights)

            filtered_heights = [highest_height_f]
            filtered_heights += [h for h in sorted_heights[1:] if abs(highest_height_f - h) < 20]

        else:
            filtered_heights = w_h

        if verbose:
            logging.info('>' * 25 + f' BORDER: filter_height_bbox' + '>' * 25)
            logging.info(f'LIST OF HEIGHT: {list_of_height}, SORTED HEIGHT: {sorted_heights}, FILTERED HEIGHTS: {filtered_heights}, AVG HEIGHT: {avg_height}')

        return filtered_heights

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
                    logging.warning(f'Skipped resizing due to invalid width: {new_width}')
            else:
                logging.warning('Skipped resizing due to invalid image height')

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
                    logging.warning(f"Image height mismatch: Resizing image from {img.shape[0]} to {min_height}")
                    img = cv2.resize(img, (img.shape[1], min_height))

                concatenated_image = cv2.hconcat([concatenated_image, color_separator, img])

            final_plate = self.process_character(concatenated_image, bg_color)

        else:
            logging.info("No valid images to merge")

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
                logging.info('=' * 20 + f' BEFORE PLATE NO: {final_string} ' + '=' * 20)
            final_string = self.match_char(final_string)
            if verbose:
                logging.info('=' * 20 + f' AFTER PLATE NO: {final_string} ' + '=' * 20)

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
                logging.info('=' * 20 + f' BEFORE PLATE NO: {final_string} ' + '=' * 20)
            final_string = self.match_char(final_string)
            if verbose:
                logging.info('=' * 20 + f' AFTER PLATE NO: {final_string} ' + '=' * 20)

            display_results(img_bgr, inv_image, img_segment, char_list, final_string, result_string, is_save=False)

            return final_string

    def process_image_folder(self, folder_path):
        image_extensions = ["*.jpg", "*.png", "*.webp"]
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

        if not image_paths:
            logging.warning("No images (.jpg, .png, .webp) found in the folder.")
            return

        for img_path in image_paths:
            logging.info(f"Processing image: {img_path}")
            self.process_image(img_path)