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
from src.Integration.arduino import Arduino
from src.config.config import config
from src.controller.matrix_controller import MatrixController
from src.controller.detection_controller import DetectionController
from src.model.gan_model import GanModel, load_rrdb_model
from src.controller.detection_controller import show_cam, check_floor, ModelAndLabelLoader
from src.model.recognize_plate.character_recognition import CharacterRecognize
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.model.cam_model import CameraV1



def main():
    db_floor = FloorController()
    IS_DEBUG = True

    if IS_DEBUG:
        # video_source = config.VIDEO_SOURCE_LAPTOP
        video_source = config.VIDEO_SOURCE_PC
        print(video_source)
        caps = [CameraV1(video, is_video=True) for video in video_source]
    else:
        video_source = config.CAM_SOURCE_LT
        caps = [CameraV1(video, is_video=False) for video in video_source]

    # Start the cameras
    for cap in caps:
        print("Starting camera: ", cap)
        cap.start()

    vehicle_model_path = config.MODEL_PATH
    vehicle_model = YOLO(vehicle_model_path)
    plate_model_path = config.MODEL_PATH_PLAT_v2
    plate_model = YOLO(plate_model_path)

    # Load the RRDB model
    gan_model = load_rrdb_model()

    char_model_path = config.MODEL_CHAR_RECOGNITION_PATH
    char_weight_path = config.WEIGHT_CHAR_RECOGNITION_PATH
    label_path = config.LABEL_CHAR_RECOGNITION_PATH

    model = ModelAndLabelLoader.load_model(char_model_path, char_weight_path)
    labels = ModelAndLabelLoader.load_labels(label_path)

    character_recognizer = CharacterRecognize(models=model, labels=labels)

    plat_detects = [None for _ in range(len(video_source))]

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
                    plat_detects[i] = DetectionController(arduino_text, matrix_total=matrix_controller, vehicle_model=vehicle_model, plate_model=plate_model, character_recognizer=character_recognizer, gan_model=gan_model)
                    plat_detects[i].start(frames[i])

                ret, frames[i] = caps[i].read()

                if not ret:
                    print(f"Failed to read frame from camera {i}")
                    continue

                # Draw bounding boxes on detected vehicles
                plat_detects[i].detect_vehicle(frames[i])

                # Process the frame for text and character recognition
                recognized_text = plat_detects[i].process_frame(frames[i])

                if recognized_text is not None:
                    print(f"Recognized Text from Camera {i}: {recognized_text}")

                # Display the frame (optional)
                show_cam(f"CAMERA: {idx} {cam_position} ", frames[i])

                # Check if 'q' key is pressed to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit key 'q' pressed. Stopping...")
                    raise KeyboardInterrupt  # Trigger shutdown

    except KeyboardInterrupt:
        print("Terminating...")

    finally:
        # Release resources
        for plat_detect in plat_detects:
            if plat_detect:
                plat_detect.stop()

        for cap in caps:
            cap.release()

        cv2.destroyAllWindows()
        print("All resources released and program terminated.")


if __name__ == "__main__":
    main()


# def main():
#     db_floor = FloorController()
#     IS_DEBUG = True

#     if IS_DEBUG:
#         video_source = config.VIDEO_SOURCE_LAPTOP
#         print(video_source)
#         caps = [CameraV1(video, is_video=True) for video in video_source]
#     else:
#         video_source = config.CAM_SOURCE_LT
#         caps = [CameraV1(video, is_video=False) for video in video_source]

#     # Start the cameras
#     for cap in caps:
#         print("Starting camera: ", cap)
#         cap.start()

#     vehicle_model_path = config.MODEL_PATH
#     vehicle_model = YOLO(vehicle_model_path)
#     plate_model_path = config.MODEL_PATH_PLAT_v2
#     plate_model = YOLO(plate_model_path)

#     char_model_path = config.MODEL_CHAR_RECOGNITION_PATH
#     char_weight_path = config.WEIGHT_CHAR_RECOGNITION_PATH
#     label_path = config.LABEL_CHAR_RECOGNITION_PATH

#     model = ModelAndLabelLoader.load_model(char_model_path, char_weight_path)
#     labels = ModelAndLabelLoader.load_labels(label_path)

#     character_recognizer = CharacterRecognize(models=model, labels=labels)

#     plat_detects = [None for _ in range(len(video_source))]

#     arduino = [Arduino(baudrate=115200, serial_number=serial) for serial in config.SERIALS]
#     frames = [None for _ in range(len(caps))]
#     total_slots = {}

#     try:
#         print("Going To Open Cam")
#         while True:
#             for i in range(len(caps)):
#                 idx, cam_position = check_floor(i)

#                 if idx not in total_slots:
#                     slot = db_floor.get_slot_by_id(idx)
#                     print(f'idx {idx}, slot {slot}')
#                     total_slot = slot["slot"]
#                     total_slots[idx] = total_slot

#                 arduino_index = idx
#                 arduino_text = arduino[arduino_index]
#                 arduino_matrix = arduino[arduino_index + 1]
#                 if plat_detects[i] is None:
#                     matrix_controller = MatrixController(arduino_matrix=arduino_matrix, max_car=18, total_car=total_slot)
#                     plat_detects[i] = DetectionController(arduino_text, matrix_total=matrix_controller, vehicle_model=vehicle_model, plate_model=plate_model, character_recognizer=character_recognizer)
#                     plat_detects[i].start(frames[i])

#                 ret, frames[i] = caps[i].read()

#                 if not ret:
#                     print(f"Failed to read frame from camera {i}")
#                     continue

#                 # Draw bounding boxes on detected vehicles
#                 plat_detects[i].detect_vehicle(frames[i])

#                 # Process the frame for text and character recognition
#                 recognized_text = plat_detects[i].process_frame(frames[i])

#                 if recognized_text is not None:
#                     print(f"Recognized Text from Camera {i}: {recognized_text}")

#                 # Display the frame (optional)
#                 # cv2.imshow(f"Video Feed - Camera {i}", frames[i])

#                 show_cam(f"CAMERA: {idx} {cam_position} ", frames[i])

#                 # Exit on 'q' key
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#     except KeyboardInterrupt:
#         print("Terminating...")

#     finally:
#         # Release resources
#         for plat_detect in plat_detects:
#             if plat_detect:
#                 plat_detect.stop()

#         for cap in caps:
#             cap.release()

#         cv2.destroyAllWindows()

