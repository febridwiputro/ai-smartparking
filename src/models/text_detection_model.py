import os
import cv2
import numpy as np
from easyocr import Reader
import time
from datetime import datetime

from src.config.config import config
from src.config.logger import logger
from src.model.recognize_plate.utils.backgrounds import check_background



def text_detection(stopped, image_restoration_result_queue, text_detection_result_queue):
    detector = TextDetector()

    while not stopped.is_set():
        try:
            restoration_result = image_restoration_result_queue.get()

            if restoration_result is None or len(restoration_result) == 0:
                continue

            restored_image = restoration_result.get("frame", None)
            bg_color = restoration_result.get("bg_color", None)

            if restored_image is None:
                continue

            text_detected_result, _ = detector.recognition_image_text(image=restored_image)

            result = {
                "bg_color": bg_color,
                "frame": text_detected_result
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
            logger.write(f"No text recognized.", logger.DEBUG)
            return []

# Text detection class using EasyOCR
class TextDetector:
    def __init__(self):
        self.ocr_net = EasyOCRNet(use_cuda=True)

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

    def filter_readtext_frame(self, texts: list, verbose=False) -> list:
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
            highest_height_f = sorted_heights[0]
            avg_height = sum(sorted_heights) / len(sorted_heights)

            filtered_heights = [highest_height_f]
            filtered_heights += [h for h in sorted_heights[1:] if abs(highest_height_f - h) < 20]

        else:
            filtered_heights = w_h

        if verbose:
            logger.write('>' * 25 + f' BORDER ' + '>' * 25, logger.DEBUG)
            logger.write(f'LIST OF HEIGHT: {list_of_height}, SORTED HEIGHT: {sorted_heights}, FILTERED HEIGHTS: {filtered_heights}, AVG HEIGHT: {avg_height}', logger.DEBUG)

        return filtered_heights 

    def save_cropped_images(self, cropped_images, save_dir="cropped_images"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for cropped_image in cropped_images:
            # Create a timestamped filename for saving the cropped image
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            image_path = os.path.join(save_dir, f"cropped_image_{timestamp}.png")
            cv2.imwrite(image_path, cropped_image)
            # print(f"Saved: {image_path}")

    def text_detect_and_recognize(self, image, is_save=False):
        bounding_boxes = self.ocr_net.detect(image)
        filtered_heights = self.filter_height_bbox(bounding_boxes=bounding_boxes)

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

        # Save the cropped images after processing them
        if cropped_images:
            if is_save:
                self.save_cropped_images(cropped_images)

        # Return cropped images and processed frame
        return cropped_images, image

    def recognition_image_text(self, image):

        bounding_boxes = self.ocr_net.readtext(image)

        cropped_images = []
        filtered_heights = self.filter_readtext_frame(bounding_boxes, False)

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

        return cropped_images, image

    def filter_height_bbox(self, bounding_boxes):
        heights = [box[3] - box[2] for group in bounding_boxes for box in group if len(box) == 4]
        return heights