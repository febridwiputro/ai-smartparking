import numpy as np
from easyocr import Reader
import time
import re

from src.config.logger import Logger
from src.utils.text_detection_util import (
    filter_height_bbox, 
    filter_readtext_frame, 
    save_cropped_images
)

logger = Logger("text_detection_model", is_save=False)

def text_detection(stopped, model_built_event, plate_result_queue, text_detection_result_queue):
    detector = TextDetector()
    model_built_event.set()

    while not stopped.is_set():
        try:
            restoration_result = plate_result_queue.get()

            if restoration_result is None or len(restoration_result) == 0:
                continue
            
            object_id = restoration_result.get("object_id")
            restored_image = restoration_result.get("frame", None)
            bg_color = restoration_result.get("bg_color", None)
            floor_id = restoration_result.get("floor_id", 0)
            cam_id = restoration_result.get("cam_id", "")
            arduino_idx = restoration_result.get("arduino_idx", None)
            car_direction = restoration_result.get("car_direction", None)
            start_line = restoration_result.get("start_line", False)
            end_line = restoration_result.get("end_line", False)
            is_centroid_inside = restoration_result.get("is_centroid_inside")

            empty_frame = np.empty((0, 0, 3), dtype=np.uint8)

            if not start_line and not end_line:
                result = {
                    "object_id": object_id,
                    "bg_color": bg_color,
                    "frame": empty_frame,
                    "plate_no_easyocr": "",
                    "floor_id": floor_id,
                    "cam_id": cam_id,
                    "arduino_idx": arduino_idx,
                    "car_direction": car_direction,
                    "start_line": start_line,
                    "end_line": end_line,
                    "is_centroid_inside": is_centroid_inside
                }
                text_detection_result_queue.put(result)
                continue

            if restored_image is None:
                continue

            text_detected_result, _, text_detected = detector.easyocr_readtext(image=restored_image)

            result = {
                "object_id": object_id,
                "bg_color": bg_color,
                "frame": text_detected_result,
                "plate_no_easyocr": text_detected,
                "floor_id": floor_id,
                "cam_id": cam_id,
                "arduino_idx": arduino_idx,
                "car_direction": car_direction,
                "start_line": start_line,
                "end_line": end_line,
                "is_centroid_inside": is_centroid_inside
            }

            text_detection_result_queue.put(result)

        except Exception as e:
            print(f"Error in text detection: {e}")
        
    del detector


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
            logger.write(f"No text recognized.", logger.DEBUG)
            return []

class TextDetector:
    def __init__(self):
        self.ocr_net = EasyOCRNet(use_cuda=True)
        self.dict_char_to_int = {'O': '0',
                            'B': '8',
                            'I': '1',
                            'J': '3',
                            'A': '4',
                            'G': '6',
                            'S': '5'}

        self.dict_int_to_char = {'0': 'O',
                            '1': 'I',
                            '8': 'O',
                            '3': 'J',
                            '4': 'A',
                            '6': 'G',
                            '5': 'S'}
        
        self.dict_3_7_to_1 = {
            '3' : "1",
            '7' : '1'
        }

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
                        logger.write(f"Skipped empty cropped image with shape: {cropped_image.shape}", logger.DEBUG)
                        continue

                    cropped_images.append(cropped_image)

        if cropped_images:
            if is_save:
                save_cropped_images(cropped_images)

        # Return cropped images and processed frame
        return cropped_images, image

    def easyocr_readtext(self, image):
        """
        Read text from the image and return cropped images, processed frame, and detected text.
        """
        bounding_boxes = self.ocr_net.readtext(image)
        cropped_images = []
        detected_texts = []
        filtered_heights = filter_readtext_frame(bounding_boxes, False)

        detect_plate = ""

        for t in bounding_boxes:
            (top_left, top_right, bottom_right, bottom_left) = t[0]
            top_left = tuple([int(val) for val in top_left])
            bottom_left = tuple([int(val) for val in bottom_left])
            top_right = tuple([int(val) for val in top_right])

            height_f = bottom_left[1] - top_left[1]
            width_f = top_right[0] - top_left[0]

            if height_f not in filtered_heights:
                continue

            top_left_y, bottom_right_y = int(max(top_left[1], 0)), int(min(bottom_right[1], image.shape[0]))
            top_left_x, bottom_right_x = int(max(top_left[0], 0)), int(min(bottom_right[0], image.shape[1]))

            cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                logger.write(f"Skipped empty cropped image with shape: {cropped_image.shape}", logger.DEBUG)
                continue

            cropped_images.append(cropped_image)

            if len(t) > 1 and t[1]:
                detected_texts.append(t[1])
                detect_plate = self.match_char(t[1])

        # print("detect_plate: ", detect_plate)

        return cropped_images, image, detect_plate

    def apply_mapping(self, text, mapping):
        """
        Replace characters in `text` based on the provided `mapping`.
        """
        return ''.join([mapping.get(char, char) for char in text])

    def match_char(self, plate):
        self.middle_char_mapping = {
            'T': '1', 
            'I': '1', 
            'D': '0', 
            'B': '8',
            'Q': '0', 
            'J': '1', 
            'Z': '7'
        }

        self.suffix_char_mapping = {
            '0': 'Q', 
            '8': 'O'
        }

        plate = re.sub(r"[^A-Za-z0-9]", "", plate)
        plate = plate.replace(" ", "").upper()

        if plate.startswith('8'):
            plate = 'B' + plate[1:]

        pattern = r"^(.{2})(.{0,4})(.*?)(.{2})$"

        def replace(match):
            prefix = match.group(1)
            middle = match.group(2)
            body = match.group(3)
            suffix = match.group(4)

            modified_middle = self.apply_mapping(middle, self.middle_char_mapping)

            if re.match(r"^[A-Z]{2}\d{4}$", f"{prefix}{modified_middle}"):
                modified_suffix = self.apply_mapping(suffix, self.suffix_char_mapping)
            else:
                modified_suffix = suffix

            modified_plate = f"{prefix}{modified_middle}{body}{modified_suffix}"
            match_special_case = re.match(r"(\d{4})(.*)(BP)$", modified_plate)
            if match_special_case:
                return f"BP{match_special_case.group(1)}{match_special_case.group(2)}"

            return modified_plate

        result = re.sub(pattern, replace, plate)
        return result

    def filter_text(self, text):
        if text == "":
            return
        text_asli = text
        
        pattern = r'[^a-zA-Z0-9]'
        text = re.sub(pattern, '', text)
        
        if len(text) >= 8:
            text = text[:8]
            
        clean_text = text.replace(" ", "").upper()
        license_plate_ = ""
        mapping = {
            0: self.dict_int_to_char,
            1: self.dict_int_to_char,
            2: self.dict_char_to_int,
            3: self.dict_char_to_int,
            4: self.dict_char_to_int,
            5: self.dict_char_to_int,
            6: self.dict_int_to_char,
            7: self.dict_int_to_char
        }

        for j in range(len(clean_text)):
            if j in mapping and clean_text[j] in mapping[j]:
                license_plate_ += mapping[j][clean_text[j]]
            else:
                license_plate_ += clean_text[j]
        
        # print(f"text_asli: {text_asli},clean_text: {clean_text}, mapping: {license_plate_}")
        if not self.filter_plat(license_plate_):
            return ""
        
        
        return license_plate_

    def filter_plat(self, text) -> bool:
      pattern = r'^BP(\d{1,4})([a-zA-Z]{1,2})$'
      match = re.match(pattern, text)
      if match:
          return True
      return False

    # def easyocr_readtext(self, image):

    #     bounding_boxes = self.ocr_net.readtext(image)

    #     cropped_images = []
    #     filtered_heights = filter_readtext_frame(bounding_boxes, False)

    #     for t in bounding_boxes:
    #         (top_left, top_right, bottom_right, bottom_left) = t[0]
    #         top_left = tuple([int(val) for val in top_left])
    #         bottom_left = tuple([int(val) for val in bottom_left])
    #         top_right = tuple([int(val) for val in top_right])

    #         height_f = bottom_left[1] - top_left[1]
    #         width_f = top_right[0] - top_left[0]

    #         # logger.write('=' * 25 + f' BORDER: EASYOCR READTEXT' + '=' * 25, logger.DEBUG)
    #         # logger.write(f'height: {height_f}, width: {width_f}', logger.DEBUG)

    #         if height_f not in filtered_heights:
    #             continue

    #         top_left_y, bottom_right_y = int(max(top_left[1], 0)), int(min(bottom_right[1], image.shape[0]))
    #         top_left_x, bottom_right_x = int(max(top_left[0], 0)), int(min(bottom_right[0], image.shape[1]))

    #         cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    #         if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
    #             logger.write(f"Skipped empty cropped image with shape: {cropped_image.shape}", logger.DEBUG)
    #             continue

    #         cropped_images.append(cropped_image)

    #     return cropped_images, image

    def filter_height_bbox(self, bounding_boxes):
        heights = [box[3] - box[2] for group in bounding_boxes for box in group if len(box) == 4]
        return heights