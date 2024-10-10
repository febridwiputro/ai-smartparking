# from paddleocr import PaddleOCR,draw_ocr
import os, sys
import re
import cv2
import time
import numpy as np
from easyocr import Reader
import logging
from src.Integration.service_v1.controller.plat_controller import PlatController
from src.model.gan_model import GanModel
from src.utils import correct_skew, find_closest_strings_dict
from src.view.show_cam import show_cam


this_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(this_path)

from model.recognize_plate.character_recognition import TextDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



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
            return results  # Returning detected texts
        else:
            logging.info("No text recognized.")
            return []


class TextRecognition:
    def __init__(self):
        self.ocr_net = EasyOCRNet(use_cuda=True)        
        self.reader = Reader(['en'], gpu=True, verbose=False)
        self.controller = PlatController()
        self.all_plat = self.controller.get_all_plat()
        self.gan_model = GanModel()
        self.td = TextDetector()        
        # self.ocr = PaddleOCR(lang='en', debug=False)
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


    def detect_and_recognize(self, image):
        """
        Detect and recognize text separately.
        """
        img_copy = image.copy()

        # Step 1: Detect text (bounding boxes)
        bounding_boxes = self.ocr_net.detect(image)
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
            logging.info('>' * 25 + f' BORDER ' + '>' * 25)
            logging.info(f'LIST OF HEIGHT: {list_of_height}, SORTED HEIGHT: {sorted_heights}, FILTERED HEIGHTS: {filtered_heights}, AVG HEIGHT: {avg_height}')

        return filtered_heights

        # # Step 2: Read text from image
        # texts = self.ocr_net.readtext(image)
        # text_bboxes = [text[0] for text in texts]
        # recognized_texts = [text[1] for text in texts]

        # # Optional: Display bounding boxes on the image
        # for bbox in text_bboxes:
        #     if len(bbox) == 4:
        #         top_left, top_right, bottom_right, bottom_left = bbox
        #         cv2.line(img_copy, tuple(map(int, top_left)), tuple(map(int, top_right)), (0, 255, 0), 2)
        #         cv2.line(img_copy, tuple(map(int, top_right)), tuple(map(int, bottom_right)), (0, 255, 0), 2)
        #         cv2.line(img_copy, tuple(map(int, bottom_right)), tuple(map(int, bottom_left)), (0, 255, 0), 2)
        #         cv2.line(img_copy, tuple(map(int, bottom_left)), tuple(map(int, top_left)), (0, 255, 0), 2)

        # # Merge text if needed
        # final_plate = self.merge_texts(recognized_texts)
        # return final_plate, text_bboxes

    def merge_texts(self, texts):
        """
        Merge recognized texts for final output.
        """
        merged_text = ''.join(texts).upper()
        return merged_text        

    def image_processing(self, image, is_bitwise=True):
        # img_correct_skew  = correct_skew(image)[1]
        # img_blur = cv2.bilateralFilter(img_correct_skew, 13, 15, 15)
        # img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        # img_resize = cv2.resize(img_gray, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        # super_res = self.gan_model.super_resolution(img_resize)
        # avg_pixel_value = np.mean(super_res)

        # if avg_pixel_value < 100 and is_bitwise:
        
        #     super_res = cv2.bitwise_not(super_res)

        # img_denoise = cv2.fastNlMeansDenoising(super_res, None, 10, 7, 21)
        # normalized_image = cv2.normalize(
        #     img_denoise, None, alpha=100, beta=250, norm_type=cv2.NORM_MINMAX)
        
    
        # show_cam("Original", normalized_image)
        # show_cam("Super Resolution", super_res)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
        resize = cv2.resize(gray, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        restorate = self.gan_model.super_resolution(resize)
        return restorate

    def recognition_image_text(self, image):
        img_copy = image.copy()
        # # param change
        texts = self.reader.readtext(image,
                                     text_threshold=0.7,
                                     low_text=0.4,
                                     decoder='greedy',
                                     slope_ths=0.6,
                                     add_margin=0.0
                                     )
        # print(self.ocr.ocr(image, cls = False)[0])
        
        if not texts:
            return "" , 0, []
        boxs = []
        text  = []
        score = []
        height = []
        rect = []
        cropped_images = []
        final_plate = ""

        filtered_heights = self.td.filter_text_frame(texts, False)

        for t in texts:
            (top_left, top_right, bottom_right, bottom_left) = t[0]
            top_left = tuple([int(val) for val in top_left])
            bottom_left = tuple([int(val) for val in bottom_left])
            top_right = tuple([int(val) for val in top_right])

            points = np.array(t[0], dtype=np.int32)
            point = cv2.boundingRect(points)
            rect.append(point)
            height.append(point[1] + point[3])
            boxs.append(t[0])
            text.append(t[1])
            score.append(t[2])

            height_f = bottom_left[1] - top_left[1]
            width_f = top_right[0] - top_left[0]
            if height_f not in filtered_heights:
                continue

            cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cropped_images.append(cropped_image)

        minimal_height = max(height)
        index_height = height.index(minimal_height)
        rect.pop(index_height)
        boxs.pop(index_height)
        # for rec in rect:
        #     cv2.rectangle(image, (rec[0], rec[1]), ((rec[0] + rec[2]), (rec[1] + rec[3])), color=(0,0,0), thickness=2)
        text.pop(index_height)
        score.pop(index_height)
        
        # TODO : slice data of text into text only
        # [([[465, 311], [527, 311], [527, 327], [465, 327]], 'BP 1062 LA', 0.17013600548741922)]
        text = ''.join(text).upper()

        if len(cropped_images) > 0:
            final_plate = self.td.process_image(cropped_images, image)

        else:
            logging.info("No valid images to merge")

        return final_plate, score, boxs
    
    def get_text_location(self, frame) -> list:
        texts = self.reader.readtext(frame, slope_ths=0.0)
        bounding = []
        for _text in texts:
            if any(char in _text[1] for char in ['0', 'O', 'Q', '8', 'D', 'o', 'q', 'd']):
                bounding.append(_text[0])
        return bounding, frame


    def filter_text(self, text, class_model = "car"):
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


# ['BP7105IT', 'BP1051FE', 'BP7105IH', 'BP1051RE', 'BP7105IR', 'BP1051RE', 'BP7105IR', 'BP1051RE', 'BP7105IR', 'BP1051RE', 'BP7105IR', 'BP1051RE', 'BP1051RE', 'BP8059RE', 'BP7105IR', 'BP7105IR', 'BP1051RE', 'BP7105IR']