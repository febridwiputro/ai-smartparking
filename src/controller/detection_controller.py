import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs (1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')  # Set TensorFlow logger to ERROR


import threading
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


def check_floor(cam_idx):
    cam_map = {
        0: (2, "IN"), 1: (2, "OUT"),
        2: (3, "IN"), 3: (3, "OUT"),
        4: (4, "IN"), 5: (4, "OUT"),
        6: (5, "IN"), 7: (5, "OUT")
    }
    return cam_map.get(cam_idx, (0, ""))

def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scale = max_width / width if width > height else max_height / height
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return image

def show_cam(text, image, max_width=1080, max_height=720):
    res_img = resize_image(image, max_width, max_height)
    cv2.imshow(text, res_img)

# Helper functions for queue management
def put_queue_none(q):
    if q is None:
        return

    try:
        q.put(None)
    except Exception as e:
        print(f"Error at putting `None` to queue: {e}")


def clear_queue(q, close=True):
    if q is None:
        return

    try:
        while not q.empty():
            try:
                q.get_nowait()
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)

    if close:
        try:
            q.close()
            q.join_thread()
        except Exception as e:
            print(e)


class VehicleDetector:
    def __init__(self, model, vehicle_queue):
        self.model = model
        self.vehicle_queue = vehicle_queue

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Received an empty image for preprocessing.")

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return image_bgr

    def predict(self, image: np.ndarray):
        preprocessed_image = self.preprocess(image)
        results = self.model.predict(preprocessed_image, conf=0.25, device="cuda:0", verbose=False, classes=config.CLASS_NAMES)

        # for result in results:
        #     print("result: ", result)
        #     # Drawing the bounding boxes on the original frame
        #     self.draw_boxes(frame=image, results=result)

        return results

    def draw_boxes(self, frame, results):
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy())
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            # label = CLASS_NAMES[cls_id]
            color = (255, 255, 255)  # Green color for bounding box
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
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
        # print("[Thread] Detecting vehicles...")
        if frame is None or frame.size == 0:
            print("Empty or invalid frame received.")
            return None

        preprocessed_image = self.preprocess(frame)
        car_frame, results = self.get_car_image(preprocessed_image)
        # self.draw_boxes(frame=frame, results=results)

        if results:
            bounding_boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if results else []
            # print(f"[Thread] Vehicle bounding boxes detected: {bounding_boxes}")
            self.vehicle_queue.put((frame, bounding_boxes))

            self.draw_box(frame=frame, boxes=bounding_boxes)

        return results

    def draw_box(self, frame, boxes):
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            color = (255, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        return frame


def plate_detection_process(stopped, vehicle_queue, plate_result_queue, model):
    plate_detector = PlateDetector(model)

    while not stopped.is_set():
        try:
            # Get the vehicle bounding boxes and the original frame
            frame, vehicle_bounding_boxes = vehicle_queue.get()

            if frame is None or not vehicle_bounding_boxes:
                print("[PlateDetection] No vehicle bounding boxes found.")
                continue

            print("[PlateDetection] Detecting plates...")
            
            # For each bounding box, crop the frame and run plate detection
            for box in vehicle_bounding_boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped_vehicle = frame[y1:y2, x1:x2]
                plate_results = plate_detector.detect_plate(cropped_vehicle)

                # Send plate detection results to the result queue
                plate_result_queue.put(plate_results)

        except Exception as e:
            print(f"Error in plate detection: {e}")



class PlateDetector:
    def __init__(self, model):
        self.model = model

        # Ensure the directory exists for saving images
        if not os.path.exists("plate_saved"):
            os.makedirs("plate_saved")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return image_bgr

    def predict(self, image: np.ndarray):
        preprocessed_image = self.preprocess(image)
        results = self.model.predict(preprocessed_image, conf=0.3, device="cuda:0", verbose=False)
        return results

    def detect_plate(self, frame):
        """
        Detect plates in the frame and return the cropped plate regions.
        Args:
            frame: The original image/frame.

        Returns:
            cropped_plates: List of cropped plate images from the frame.
        """
        # Detect plates and get bounding boxes
        results = self.predict(frame)

        if not results:
            print("[PlateDetector] No plates detected.")
            return []

        bounding_boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if results else []
        if not bounding_boxes:
            print("[PlateDetector] No bounding boxes found.")
            return []

        print(f"[PlateDetector] Bounding boxes detected: {bounding_boxes}")

        # Extract and return cropped plates based on bounding boxes
        cropped_plates = self.get_cropped_plates(frame, bounding_boxes)

        # Save cropped plates by passing both the frame and the bounding boxes
        self.save_cropped_plate(frame, bounding_boxes)

        return cropped_plates

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
            # Ensure coordinates are within the frame dimensions
            x1, y1, x2, y2 = [max(0, min(int(coord), width if i % 2 == 0 else height)) for i, coord in enumerate(box)]

            # Crop the detected plate region from the frame
            cropped_plate = frame[y1:y2, x1:x2]
            
            # Only add the plate if the cropped region is not empty
            if cropped_plate.size > 0:
                cropped_plates.append(cropped_plate)

        return cropped_plates

    def save_cropped_plate(self, frame, boxes):
        """
        Save the cropped plate regions as image files.
        Args:
            frame: The original image/frame.
            boxes: List of bounding boxes (each box is [x1, y1, x2, y2]).
        """
        import os
        from datetime import datetime

        if not os.path.exists('plate_saved'):
            os.makedirs('plate_saved')

        for i, box in enumerate(boxes):
            # Extract coordinates from the bounding box
            x1, y1, x2, y2 = map(int, box)

            # Ensure valid crop area
            cropped_plate = frame[y1:y2, x1:x2]
            if cropped_plate.size > 0:
                # Create a filename with the current timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                filename = f'plate_saved/{timestamp}.jpg'

                # Save the cropped plate image
                cv2.imwrite(filename, cropped_plate)
                print(f"[PlateDetector] Saved cropped plate as {filename}")



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

# Image restoration using GAN model (Multiprocessing)
def image_restoration(stopped, plate_result_queue, img_restoration_result_queue, gan_model):
    while not stopped.is_set():
        try:
            print("[Process] Restoring image...")
            image = plate_result_queue.get()

            # Ensure the image is valid and check if it's a NumPy array
            if image is None or not isinstance(image, np.ndarray):
                print("[Process] Invalid image received.")
                continue

            # Convert to grayscale and resize
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

            # Perform super resolution
            restored_image = gan_model.super_resolution(resized_image)

            print("[Process] Image restored")
            # Put the restored image back in the queue for further processing
            img_restoration_result_queue.put(restored_image)

        except Exception as e:
            print(f"Error in image_restoration: {e}")


# Text detection using EasyOCR (Multiprocessing)
def text_detection(stopped, frame_queue, result_queue):
    detector = TextDetector()

    while not stopped.is_set():
        try:
            print("[Process] Detecting text...")
            image = frame_queue.get()
            if image is None:
                continue

            # Perform text detection and recognition
            cropped_images, processed_frame = detector.text_detect_and_recognize(image)

            print("[Process] Text detection complete")
            # Put the processed frame with detected text bounding boxes in the result queue
            result_queue.put((cropped_images, processed_frame))

        except Exception as e:
            print(f"Error in text detection: {e}")

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

def character_recognition(stopped, cropped_queue, result_queue, character_recognizer):
    """
    Function to load the model within the process and perform character recognition.
    """
    cr = character_recognizer

    while not stopped.is_set():
        try:
            cropped_images = cropped_queue.get()
            if cropped_images is None:
                continue

            # Process the cropped images (containing text)
            plate_text = cr.process_image(cropped_images, None)

            # Put the recognized text into the result queue
            result_queue.put(plate_text)

        except Exception as e:
            print(f"Error in character recognition: {e}")


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

    def filter_readtext_frame(self, texts: list, verbose=False) -> str:
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
            logging.info('>' * 25 + f' BORDER ' + '>' * 25)
            logging.info(f'LIST OF HEIGHT: {list_of_height}, SORTED HEIGHT: {sorted_heights}, FILTERED HEIGHTS: {filtered_heights}, AVG HEIGHT: {avg_height}')


    def text_detect_and_recognize(self, image):
        """
        Detect and recognize text separately.
        """
        bounding_boxes = self.ocr_net.detect(image)

        filtered_heights = self.filter_height_bbox(bounding_boxes=bounding_boxes)

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
        
        return cropped_images, image


    def recognition_image_text(self, image):

        bounding_boxes = self.ocr_net.detect(image)

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

class DetectionController:
    def __init__(self, arduino_matrix, matrix_total, vehicle_model, plate_model, character_recognizer, gan_model):
        self.matrix_text = MatrixController(arduino_matrix, 0, 100)
        self.matrix_text.start()
        self.matrix = matrix_total
        self.matrix.start(self.matrix.get_total())
        self.plate_model_path = plate_model

        self.vehicle_queue = mp.Queue()
        self.plate_result_queue = mp.Queue()
        self.img_restoration_result_queue = mp.Queue()

        self.vehicle_detector = VehicleDetector(vehicle_model, self.vehicle_queue)
        self.stopped = mp.Event()

        self.frame_queue = mp.Queue()
        self.text_result_queue = mp.Queue()
        self.cropped_queue = mp.Queue()
        self.char_result_queue = mp.Queue()

        self.gan_model = GanModel(gan_model)

        self.character_recognizer = character_recognizer

        self.vehicle_thread = None
        self.plate_detection_process = None
        self.vehicle_bounding_boxes = []

    def start(self, frame):
        print("[Thread] Starting vehicle detection thread...")
        self.vehicle_thread = threading.Thread(target=self.detect_vehicle, args=(frame,))
        self.vehicle_thread.start()

        print("[Process] Starting plate detection process...")
        self.plate_detection_process = mp.Process(target=plate_detection_process, args=(self.stopped, self.vehicle_queue, self.plate_result_queue, self.plate_model_path))
        self.plate_detection_process.start()

        print("[Process] Starting image restoration process...")
        self.image_restoration_process = mp.Process(target=image_restoration, args=(self.stopped, self.plate_result_queue, self.img_restoration_result_queue, self.gan_model))
        self.image_restoration_process.start()

        # print("[Process] Starting text detection process...")
        # self.text_detection_process = mp.Process(target=text_detection, args=(self.stopped, self.frame_queue, self.text_result_queue))
        # self.text_detection_process.start()

        # print("[Process] Starting character recognition process...")
        # self.char_recognition_process = mp.Process(target=character_recognition, args=(self.stopped, self.cropped_queue, self.char_result_queue, self.character_recognizer))
        # self.char_recognition_process.start()

    def detect_vehicle(self, frame):
        # print("[Thread] Detecting vehicles...")
        vehicle_results = self.vehicle_detector.detect_vehicle(frame)
        if vehicle_results:
            self.vehicle_bounding_boxes = vehicle_results[0].boxes.xyxy.cpu().numpy().tolist() if vehicle_results else []
            # print(f"Vehicle bounding boxes: {self.vehicle_bounding_boxes}")

    def stop(self):
        print("[Controller] Stopping detection processes and threads...")
        self.stopped.set()

        put_queue_none(self.frame_queue)
        put_queue_none(self.img_restoration_result_queue)
        # put_queue_none(self.text_result_queue)
        put_queue_none(self.cropped_queue)
        # put_queue_none(self.char_result_queue)
        put_queue_none(self.vehicle_queue)
        put_queue_none(self.plate_result_queue)

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

    #     # if self.text_detection_process is not None:
    #     #     self.text_detection_process.join()
    #     #     self.text_detection_process = None

    #     # if self.char_recognition_process is not None:
    #     #     self.char_recognition_process.join()
    #     #     self.char_recognition_process = None

        # Clear all queues
        clear_queue(self.frame_queue)
        clear_queue(self.img_restoration_result_queue)
        # clear_queue(self.text_result_queue)
        clear_queue(self.cropped_queue)
        # clear_queue(self.char_result_queue)
        clear_queue(self.vehicle_queue)
        clear_queue(self.plate_result_queue)

        print("[Controller] All processes and threads stopped.")

    def process_frame(self, frame):
        # print("[Thread] Processing frame for text detection...")
        self.frame_queue.put(frame)

        # Get the detected text regions
        if not self.text_result_queue.empty():
            cropped_images, _ = self.text_result_queue.get()

            # Send cropped text regions for character recognition
            print("[Process] Sending cropped text regions for character recognition...")
            self.cropped_queue.put(cropped_images)

            # Get recognized text from character recognition
            if not self.char_result_queue.empty():
                recognized_text = self.char_result_queue.get()
                print(f"[Process] Recognized text: {recognized_text}")
                return recognized_text

        return None
