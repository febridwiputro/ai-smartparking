import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs (1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

import threading
import queue
import multiprocessing as mp
import cv2
import numpy as np
from ultralytics import YOLO
from easyocr import Reader
import time
import logging
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

from src.config.config import config
from src.config.logger import logger
from src.controller.matrix_controller import MatrixController
from src.model.gan_model import GanModel
from src.model.recognize_plate.character_recognition import CharacterRecognize
from utils.multiprocessing_util import put_queue_none, clear_queue, check_floor, resize_image, show_cam
from src.model.recognize_plate.utils.backgrounds import check_background


class VehicleDetector:
    def __init__(self, model, vehicle_result_queue):
        self.model = model
        self.vehicle_result_queue = vehicle_result_queue

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Received an empty image for preprocessing.")

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return image_bgr

    def predict(self, image: np.ndarray):
        preprocessed_image = self.preprocess(image)
        results = self.model.predict(preprocessed_image, conf=0.25, device="cuda:0", verbose=False, classes=config.CLASS_NAMES)
        return results

    def draw_boxes(self, frame, results):
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy())
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            color = (255, 255, 255)  # White color for bounding box
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot
        
        return frame

    def get_car_image(self, frame, threshold=0.008):
        results = self.predict(frame)
        if not results[0].boxes.xyxy.cpu().tolist():
            return np.array([]), results
        boxes = results[0].boxes.xyxy.cpu().tolist()
        # print("boxes : ", boxes)
        height, width = frame.shape[:2]
        filtered_boxes = [box for box in boxes if (box[3] < height * (1 - threshold))]
        if not filtered_boxes:
            return np.array([]), results
        sorted_boxes = sorted(filtered_boxes, key=lambda x: x[3] - x[1], reverse=True)
        if len(sorted_boxes) > 0:
            box = sorted_boxes[0]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            car_frame = frame[y1:y2, x1:x2]
            if car_frame.shape[0] == 0 or car_frame.shape[1] == 0:
                return np.array([]), results
            return car_frame, results
        return np.array([]), results

    def detect_vehicle(self, frame):
        if frame is None or frame.size == 0:
            print("Empty or invalid frame received.")
            return None

        # Preprocess frame and get the vehicle frame (cropped)
        preprocessed_image = self.preprocess(frame)
        car_frame, results = self.get_car_image(preprocessed_image)

        if car_frame.size > 0:
            self.vehicle_result_queue.put(car_frame)

            # Extract bounding boxes from results
            boxes = results[0].boxes.xyxy.cpu().tolist()

            # Draw boxes on the original frame
            self.draw_box(frame=frame, boxes=boxes)

        return car_frame

    def draw_box(self, frame, boxes):
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            color = (255, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        return frame


def plate_detection_process(stopped, vehicle_queue, plate_result_queue):
    plate_model_path = config.MODEL_PATH_PLAT_v2
    plate_model = YOLO(plate_model_path)
    plate_detector = PlateDetector(plate_model)

    while not stopped.is_set():
        try:
            frame = vehicle_queue.get()

            if frame is None:
                continue
            
            plate_results = plate_detector.detect_plate(frame)

            if plate_results:
                # print("plate_results: ", plate_results)
                plate_result_queue.put(plate_results)

        except Exception as e:
            print(f"Error in plate detection: {e}")

class PlateDetector:
    def __init__(self, model):
        self.model = model

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return image_bgr

    def predict(self, image: np.ndarray):
        preprocessed_image = self.preprocess(image)
        results = self.model.predict(preprocessed_image, conf=0.3, device="cuda:0", verbose=False)
        return results

    def detect_plate(self, frame, is_save=True):
        results = self.predict(frame)

        if not results:
            print("[PlateDetector] No plates detected.")
            return []

        bounding_boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if results else []
        if not bounding_boxes:
            return []

        cropped_plates = self.get_cropped_plates(frame, bounding_boxes)

        if is_save:
            self.save_cropped_plate(cropped_plates)

        return cropped_plates

    def save_cropped_plate(self, cropped_plates):
        """
        Save the cropped plate regions as image files.
        Args:
            cropped_plates: List of cropped plate images.
        """
        import os
        from datetime import datetime

        if not os.path.exists('plate_saved'):
            os.makedirs('plate_saved')

        for i, cropped_plate in enumerate(cropped_plates):
            if cropped_plate.size > 0:
                # Create a filename with the current timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                filename = f'plate_saved/{timestamp}.jpg'

                # Save the cropped plate image
                cv2.imwrite(filename, cropped_plate)

    def is_valid_cropped_plate(self, cropped_plate):
        """Check if the cropped plate meets the size requirements."""
        height, width = cropped_plate.shape[:2]
        if height < 55 or width < 100:
            return False
        if height >= width:
            return False
        compare = abs(height - width)
        if compare <= 100 or compare >= 400:
            return False
        return True

    def get_cropped_plates(self, frame, boxes):
        """
        Extract cropped plate images based on bounding boxes.
        Args:
            frame: The original image/frame.
            boxes: List of bounding boxes (each box is [x1, y1, x2, y2]).

        Returns:
            cropped_plates: List of cropped plate images.
        """
        height, width, _ = frame.shape
        cropped_plates = []

        for box in boxes:
            x1, y1, x2, y2 = [max(0, min(int(coord), width if i % 2 == 0 else height)) for i, coord in enumerate(box)]
            cropped_plate = frame[y1:y2, x1:x2]

            if cropped_plate.size > 0 and self.is_valid_cropped_plate(cropped_plate):
                cropped_plates.append(cropped_plate)

        return cropped_plates


    def draw_boxes(self, frame, boxes):
        """
        Draw bounding boxes for detected plates on the frame.
        Args:
            frame: The original image/frame.
            boxes: List of bounding boxes to draw (each box is [x1, y1, x2, y2]).
        """
        height, width, _ = frame.shape

        for box in boxes:
            # Ensure coordinates are within the frame dimensions
            x1, y1, x2, y2 = [max(0, min(int(coord), width if i % 2 == 0 else height)) for i, coord in enumerate(box)]

            # Draw the bounding box on the frame
            color = (0, 255, 0)  # Green for plate bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw the center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot in the center

        return frame


def image_restoration(stopped, plate_result_queue, img_restoration_result_queue):
    img_restore = ImageRestoration()

    while not stopped.is_set():
        try:
            plate_result = plate_result_queue.get()
            if plate_result is None or len(plate_result) == 0:
                continue

            plate_image = plate_result[0]

            if plate_image is None:
                continue

            restored_image = img_restore.process_image(plate_image)
            img_restoration_result_queue.put(restored_image)

        except Exception as e:
            print(f"Error in image_restoration: {e}")


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


def text_detection(stopped, image_restoration_result_queue, text_detection_result_queue):
    detector = TextDetector()

    while not stopped.is_set():
        # try:
        # print("[Process] Detecting text...")
        image = image_restoration_result_queue.get()
        if image is None:
            continue

        bg_status = check_background(image, False)

        # print("bg_status: ", bg_status)

        # Perform text detection and recognition
        # text_detected_result, _ = detector.text_detect_and_recognize(image)
        text_detected_result, _ = detector.recognition_image_text(image=image)

        # print("[Process] Text detection complete")
        # print("text_detected_result: ", text_detected_result)

        # Prepare result as a dictionary
        result = {
            "bg_status": bg_status,  # e.g., "bg_white"
            "text_detection_frame": text_detected_result
        }

        # Put the result into the queue
        text_detection_result_queue.put(result)

        # except Exception as e:
        #     print(f"Error in text detection: {e}")


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
            logging.info('>' * 25 + f' BORDER: FILTER READTEXT FRAME ' + '>' * 25)
            logging.info(f'LIST OF HEIGHT: {list_of_height}, SORTED HEIGHT: {sorted_heights}, FILTERED HEIGHTS: {filtered_heights}, AVG HEIGHT: {avg_height}')

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
                        logging.warning(f"Skipped empty cropped image with shape: {cropped_image.shape}")
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
                logging.warning(f"Skipped empty cropped image with shape: {cropped_image.shape}")
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
            print(f'Model char_recognize loaded')
            # print(f'Model loaded from {model_path} and weights from {weight_path}')
            return model
        except Exception as e:
            print(f'Could not load model: {e}')
            return None

    @staticmethod
    def load_labels(labels_path):
        """Loads labels using LabelEncoder from a NumPy file."""
        try:
            labels = LabelEncoder()
            labels.classes_ = np.load(labels_path)
            print(f'Label char_recognize loaded')
            # print(f'Labels loaded from {labels_path}')
            return labels
        except Exception as e:
            print(f'Could not load labels: {e}')
            return None

def character_recognition(stopped, text_detection_result_queue, char_recognize_result_queue):
    """
    Function to load the model within the process and perform character recognition.
    """
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

            # Parse bg_status and cropped_images from the result
            bg_status = result.get("bg_status")
            cropped_images = result.get("text_detection_frame")

            plate_text = cr.process_image(cropped_images, bg_status)
            print("plate_text: ", plate_text)

            # Put the recognized text into the result queue
            char_recognize_result_queue.put({
                "bg_status": bg_status,
                "recognized_text": plate_text
            })

        except Exception as e:
            print(f"Error in character recognition: {e}")


class DetectionController:
    def __init__(self, arduino_matrix, matrix_total):
        self.matrix_text = MatrixController(arduino_matrix, 0, 100)
        self.matrix_text.start()
        self.matrix = matrix_total
        self.matrix.start(self.matrix.get_total())
        self.vehicle_result_queue = mp.Queue()
        self.plate_result_queue = mp.Queue()
        self.img_restoration_result_queue = mp.Queue()
        self.text_detection_result_queue = mp.Queue()
        self.stopped = mp.Event()
        self.char_recognize_result_queue = mp.Queue()
        self.vehicle_thread = None
        self.plate_detection_process = None
        self.vehicle_bounding_boxes = []
        self._current_frame = None
        self._current_result = None

    def start(self):
        print("[Thread] Starting vehicle detection thread...")
        self.vehicle_thread = threading.Thread(target=self.detect_vehicle_work_thread)
        self.vehicle_thread.start()

        print("[Process] Starting plate detection process...")
        self.plate_detection_process = mp.Process(target=plate_detection_process, args=(self.stopped, self.vehicle_result_queue, self.plate_result_queue))
        self.plate_detection_process.start()

        # print("[Process] Starting image restoration process...")
        self.image_restoration_process = mp.Process(target=image_restoration, args=(self.stopped, self.plate_result_queue, self.img_restoration_result_queue))
        self.image_restoration_process.start()

        # print("[Process] Starting text detection process...")
        self.text_detection_process = mp.Process(target=text_detection, args=(self.stopped, self.img_restoration_result_queue, self.text_detection_result_queue))
        self.text_detection_process.start()

        # print("[Process] Starting character recognition process...")
        self.char_recognition_process = mp.Process(target=character_recognition, args=(self.stopped, self.text_detection_result_queue, self.char_recognize_result_queue))
        self.char_recognition_process.start()
    
    def process_frame(self, frame):
        self._current_frame = frame

    def get_results(self):
        return self._current_result
        # get dari result terakhir

    def detect_vehicle_work_thread(self):
        # TODO define YOLO MODEL
        vehicle_model = YOLO(config.MODEL_PATH)
        vehicle_detector = VehicleDetector(vehicle_model, self.vehicle_result_queue)
        
        while True:
            if self.stopped.is_set():
                break

            if self._current_frame is None or self._current_frame.size == 0:
                continue

            frame = self._current_frame.copy()

            if frame is None or frame.size == 0:
                print("Empty or invalid frame received.")
                continue
            
            # self.car_detection_result = result dari yolo
            # TODO model detect disimi
            # put cropped car
            # pakai try except
            try:
                if frame is None or frame.size == 0:
                    print("Empty or invalid frame received.")
                    return None

                vehicle_detector.detect_vehicle(frame=frame)

            except Exception as e:
                print(f"Error in vehicle_detector: {e}")
    

    def post_process_work_thread(self):
        while True:
            result = self.char_recognize_result_queue.get()
            # print("char_recognize_result_queue: ", result)
            # process result disini
            self._current_result = result

    def stop(self):
        print("[Controller] Stopping detection processes and threads...")
        self.stopped.set()

        put_queue_none(self.vehicle_result_queue)
        put_queue_none(self.plate_result_queue)
        put_queue_none(self.img_restoration_result_queue)
        put_queue_none(self.text_detection_result_queue)
        put_queue_none(self.char_recognize_result_queue)

        # Stop threads
        if self.vehicle_thread is not None:
            self.vehicle_thread.join()
            self.vehicle_thread = None

        # Stop processes
        if self.plate_detection_process is not None:
            self.plate_detection_process.join()
            self.plate_detection_process = None

        if self.image_restoration_process is not None:
            self.image_restoration_process.join()
            self.image_restoration_process = None

        if self.text_detection_process is not None:
            self.text_detection_process.join()
            self.text_detection_process = None

        if self.char_recognition_process is not None:
            self.char_recognition_process.join()
            self.char_recognition_process = None

        # Clear all queues
        clear_queue(self.vehicle_result_queue)
        clear_queue(self.plate_result_queue)
        clear_queue(self.img_restoration_result_queue)
        clear_queue(self.text_detection_result_queue)
        clear_queue(self.char_recognize_result_queue)


        print("[Controller] All processes and threads stopped.")