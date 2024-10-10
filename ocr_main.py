import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs (1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')  # Set TensorFlow logger to ERROR

import argparse
import threading
import time
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from src.Integration.arduino import Arduino
from src.config.config import config
from src.controller.matrix_controller import MatrixController
from src.view.show_cam import show_cam
from src.controller.ocr_controller import OCRController
from src.controller.ocr_controller_mp import OCRControllerMP
from src.model.recognize_plate.character_recognition import CharacterRecognize
from src.model.cam_model import CameraV1
from src.model.matrix_model import CarCounterMatrix
from src.Integration.service_v1.controller.floor_controller import FloorController
from ultralytics import YOLO

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

# Helper function to map camera index to floor and direction
def check_floor(cam_idx):
    cam_map = {
        0: (2, "IN"), 1: (2, "OUT"),
        2: (3, "IN"), 3: (3, "OUT"),
        4: (4, "IN"), 5: (4, "OUT"),
        6: (5, "IN"), 7: (5, "OUT")
    }
    return cam_map.get(cam_idx, (0, ""))

def main():
    db_floor = FloorController()
    IS_DEBUG = True
    IS_MP = False

    try:
        print("Loading model...")
        VEHICLE_DETECTION_MODEL = YOLO(config.MODEL_PATH)
        PLATE_DETECTION_MODEL = YOLO(config.MODEL_PATH_PLAT_v2)

        model = ModelAndLabelLoader.load_model(config.MODEL_CHAR_RECOGNITION_PATH, config.WEIGHT_CHAR_RECOGNITION_PATH)
        labels = ModelAndLabelLoader.load_labels(config.LABEL_CHAR_RECOGNITION_PATH)

        if model is None or labels is None:
            print("Failed to load model or labels, exiting.")
            return

        print("Model loaded.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    CHARACTER_RECOGNITION = CharacterRecognize(models=model, labels=labels)

    if IS_DEBUG:
        video_source = config.VIDEO_SOURCE_LAPTOP
        # video_source = config.VIDEO_SOURCE_PC
        # video_source = config.VIDEO_SOURCE_20241004
        print(video_source)
        caps = [CameraV1(video, is_video=True) for video in video_source]
    else:
        video_source = config.CAM_SOURCE_LT
        caps = [CameraV1(video, is_video=False) for video in video_source]

    plat_detects = [None for _ in range(len(video_source))]

    for cap in caps:
        print("Starting camera: ", cap)
        cap.start()

    arduino = [Arduino(baudrate=115200, serial_number=serial) for serial in config.SERIALS]
    frames = [None for _ in range(len(caps))]
    total_slots = {}

    try:
        print("Going To Open Cam")
        while True:
            for i in range(len(caps)):

                idx, cam_position = check_floor(i)

                if idx not in total_slots:
                    slot = db_floor.get_slot_by_id(idx)
                    print(f'idx {idx}, slot {slot}')
                    total_slot = slot["slot"]
                    total_slots[idx] = total_slot

                arduino_index = idx
                arduino_text = arduino[arduino_index]
                arduino_matrix = arduino[arduino_index + 1]
                if plat_detects[i] is None:
                    matrix_controller = MatrixController(arduino_matrix=arduino_matrix, max_car=18, total_car=total_slot)
                    
                    # Conditional to use multiprocessing or not
                    if IS_MP:
                        # Using multiprocessing
                        plat_detects[i] = OCRControllerMP(arduino_text, matrix_total=matrix_controller, yolo_model=VEHICLE_DETECTION_MODEL, character_recognition=CHARACTER_RECOGNITION)
                        print(f"Multiprocessing enabled for camera {i}.")
                    else:
                        # Without multiprocessing
                        plat_detects[i] = OCRController(arduino_text, matrix_total=matrix_controller, vehicle_detection_model=VEHICLE_DETECTION_MODEL, character_recognition=CHARACTER_RECOGNITION, plate_detection_model=PLATE_DETECTION_MODEL)
                        # plat_detects[i] = OCRController(arduino_text, matrix_total=matrix_controller, yolo_model=yolo_model)
                        print(f"Multiprocessing disabled for camera {i}.")

                _, frames[i] = caps[i].read()
                # frames[i] = cv2.resize(frames[i], (1080, 720))
                if frames[i] is not None:
                    vehicle_detected = plat_detects[i].car_direct(frames[i], arduino_idx=arduino_text, cam_idx=i)

                    if vehicle_detected:
                        plat_detects[i].processing_car_counter(vehicle_detected)
                        plat_detects[i].processing_ocr(arduino_text, i, frames[i], vehicle_detected)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 32:
                cv2.waitKey(0)
        
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
    
    except KeyboardInterrupt:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        print("KeyboardInterrupt")
        exit(0)


if __name__ == "__main__":
    main()