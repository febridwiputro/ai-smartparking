import os
import time
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs (1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

import threading
import multiprocessing as mp
import cv2
import numpy as np
from ultralytics import YOLO

from src.config.config import config
from src.config.logger import logger
from src.models.vehicle_plate_model_v7 import VehicleDetector
from utils.multiprocessing_util import put_queue_none, clear_queue

from src.controllers.utils.util import (
    define_tracking_polygon
)


class DetectionControllerV7:
    def __init__(self, vehicle_plate_result_queue=None, arduino_matrix=None):
        self.arduino_matrix = arduino_matrix
        self.vehicle_plate_result_queue = vehicle_plate_result_queue
        self.stopped = mp.Event()
        self.vehicle_thread = None
        self._current_frame = None
        self.lock_frame = threading.Lock()
        self._model_built_event = mp.Event()

    def start(self):
        print("[Thread] Starting vehicle detection thread...")
        self.vehicle_thread = threading.Thread(target=self.detect_vehicle_work_thread)
        self.vehicle_thread.start()

    def process_frame(self, frame_bundle: dict):
        self._current_frame = frame_bundle

    def detect_vehicle_work_thread(self):
        vehicle_plate_model = YOLO(config.MODEL_VEHICLE_PLATE_PATH)
        vehicle_detector = VehicleDetector(vehicle_plate_model, is_vehicle_model=False)
        self._model_built_event.set()

        prev_qsize = None

        while True:
            if self.stopped.is_set():
                break
            
            if self._current_frame is None or len(self._current_frame) == 0:
                print("Empty or invalid frame received.")
                time.sleep(0.1)
                continue

            frame_bundle = self._current_frame.copy()

            frame = frame_bundle["frame"]
            floor_id = frame_bundle["floor_id"]
            cam_id = frame_bundle["cam_id"]

            if frame is None or frame.size == 0:
                print("Empty or invalid frame received.")
                time.sleep(0.1)
                continue

            try:
                height, width = frame.shape[:2]

                poly_points, tracking_points, poly_bbox = define_tracking_polygon(
                    height=height, width=width, 
                    floor_id=floor_id, cam_id=cam_id
                )

                vehicle_plate_data, cropped_frame, is_centroid_inside, car_info = vehicle_detector.vehicle_detect(arduino_idx=str(self.arduino_matrix), frame=frame, floor_id=floor_id, cam_id=cam_id, tracking_points=tracking_points, poly_bbox=poly_bbox)

                if vehicle_plate_data is not None and isinstance(vehicle_plate_data, dict):
                    if self.vehicle_plate_result_queue is not None:
                        current_qsize = self.vehicle_plate_result_queue.qsize()

                        if current_qsize != 0 or current_qsize == 1:
                            if current_qsize != prev_qsize:
                                print("q_size: ", current_qsize)
                                prev_qsize = current_qsize

                        # print("vehicle_plate_data 2: ", vehicle_plate_data)

                        self.vehicle_plate_result_queue.put(vehicle_plate_data)

            except Exception as e:
                print(f"Error in vehicle_detector: {e}")

    def is_model_built(self):
        return self._model_built_event.is_set()

    def stop(self):
        print("[Controller] Stopping detection processes and threads...")
        self.stopped.set()

        put_queue_none(self.vehicle_plate_result_queue)

        # Stop threads
        if self.vehicle_thread is not None:
            self.vehicle_thread.join()
            self.vehicle_thread = None

        # Clear all queues
        clear_queue(self.vehicle_plate_result_queue)

        print("[Controller] All processes and threads stopped.")
