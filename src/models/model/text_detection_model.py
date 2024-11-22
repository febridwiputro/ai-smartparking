# from paddleocr import PaddleOCR,draw_ocr
import os, sys
import re
import cv2
import time
import numpy as np
from easyocr import Reader
import logging
import multiprocessing as mp
from src.Integration.service_v1.controller.plat_controller import PlatController
from src.model.gan_model import GanModel
from src.utils import correct_skew, find_closest_strings_dict
from src.view.show_cam import show_cam


this_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(this_path)

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

def _restoration_process_image(image_queue: mp.Queue, result_queue: mp.Queue, gan_model: GanModel, stop_event: mp.Event):
    while not stop_event.is_set():
        try:
            image = image_queue.get(timeout=5)
            if image is None:
                continue
            
            print("Image received for restoration")  # Log tambahan
            
            restored_image = gan_model.super_resolution(image)
            result_queue.put(restored_image)
            
            print("Restored image added to result queue")  # Log tambahan
        except mp.queues.Empty:
            pass
        except Exception as e:
            print(f"Error in restoration process: {e}")


def display_images(result_queue: mp.Queue):
    """Function to display images in the main process."""
    try:
        while True:
            restored_image = result_queue.get(timeout=5)
            if restored_image is None:
                break

            # Display the processed image
            cv2.imshow("restored_image", restored_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error in display_images: {e}")
    finally:
        cv2.destroyAllWindows()

class TextDetector:
    def __init__(self, character_recognition):
        self.ocr_net = EasyOCRNet(use_cuda=True)        
        self.reader = Reader(['en'], gpu=True, verbose=False)
        self.controller = PlatController()
        self.all_plat = self.controller.get_all_plat()
        self.gan_model = GanModel()
        self.cr = character_recognition  
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
    
        self.image_queue = None
        self.result_queue = None
        self.stop_event = mp.Event()
        self.image_queue = None
        self.result_queue = None
        self.stop_event = mp.Event()    

    def start(self):
        self.image_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.stop_event = mp.Event()
        self.image_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.stop_event = mp.Event()

        self.process = mp.Process(target=_restoration_process_image, args=(self.image_queue, self.result_queue, self.gan_model, self.stop_event))
        self.process.start()

    def image_processing(self, image: np.ndarray, is_bitwise=True) -> np.ndarray:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resize = cv2.resize(gray, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

            self.image_queue.put(resize)
            
            try:
                restored_image = self.result_queue.get(timeout=10)  # Tambah timeout atau gunakan retry mekanisme
            except mp.queues.Empty:
                print("Timeout while waiting for restored image.")
                return resize  # Fallback ke gambar resized jika timeout

            return restored_image
        except Exception as e:
            print(f"Error in image_processing: {e}")
            return np.array([])

    def stop(self):
        """Stops the multiprocessing process cleanly."""
        self.stop_event.set()
        self.image_queue.put(None)
        self.process.join()

    def image_restoration(self, image, is_bitwise=True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
        resize = cv2.resize(gray, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        restorate = self.gan_model.super_resolution(resize)
        return restorate

    def text_detect_and_recognize(self, image):
        """
        Detect and recognize text separately.
        """
        img_copy = image.copy()
        bounding_boxes = self.ocr_net.detect(image)

        filtered_heights = self.cr.filter_height_bbox(bounding_boxes=bounding_boxes)

        converted_bboxes = []
        cropped_images = []
        final_plate = ""
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

                    if height_bbox not in filtered_heights:
                        continue
    
                    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                        logging.warning(f"Skipped empty cropped image with shape: {cropped_image.shape}")
                        continue

                    cropped_images.append(cropped_image)

        if len(cropped_images) > 0:

            print("cropped_images: ", cropped_images)
            final_plate = self.cr.process_image(cropped_images, image)

        else:
            logging.info("No valid images to merge")

        return final_plate

    def recognition_image_text(self, image):
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

        filtered_heights = self.cr.filter_text_frame(texts, False)

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

            top_left_y, bottom_right_y = int(max(top_left[1], 0)), int(min(bottom_right[1], image.shape[0]))
            top_left_x, bottom_right_x = int(max(top_left[0], 0)), int(min(bottom_right[0], image.shape[1]))

            cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                logging.warning(f"Skipped empty cropped image with shape: {cropped_image.shape}")
                continue

            cropped_images.append(cropped_image)

            # cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            # cropped_images.append(cropped_image)

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
            final_plate = self.cr.process_image(cropped_images, image)

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