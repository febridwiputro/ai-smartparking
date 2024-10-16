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

def text_detection(stopped, plate_result_queue, text_detection_result_queue):
    detector = TextDetector()

    while not stopped.is_set():
        try:
            restoration_result = plate_result_queue.get()

            if restoration_result is None or len(restoration_result) == 0:
                continue

            restored_image = restoration_result.get("frame", None)
            bg_color = restoration_result.get("bg_color", None)
            floor_id = restoration_result.get("floor_id", 0)
            cam_id = restoration_result.get("cam_id", "")
            arduino_idx = restoration_result.get("arduino_idx", None)
            car_direction = restoration_result.get("car_direction", None)
            start_line = restoration_result.get("start_line", False)
            end_line = restoration_result.get("end_line", False)

            empty_frame = np.empty((0, 0, 3), dtype=np.uint8)

            if not start_line and not end_line:
                result = {
                    "bg_color": bg_color,
                    "frame": empty_frame,
                    "floor_id": floor_id,
                    "cam_id": cam_id,
                    "arduino_idx": arduino_idx,
                    "car_direction": car_direction,
                    "start_line": start_line,
                    "end_line": end_line
                }
                text_detection_result_queue.put(result)
                continue

            if restored_image is None:
                continue

            text_detected_result, _ = detector.easyocr_readtext(image=restored_image)

            result = {
                "bg_color": bg_color,
                "frame": text_detected_result,
                "floor_id": floor_id,
                "cam_id": cam_id,
                "arduino_idx": arduino_idx,
                "car_direction": car_direction,
                "start_line": start_line,
                "end_line": end_line
            }

            text_detection_result_queue.put(result)

        except Exception as e:
            print(f"Error in text detection: {e}")


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