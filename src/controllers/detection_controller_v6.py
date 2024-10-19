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
from src.models.vehicle_plate_model_v6 import VehicleDetector
from utils.multiprocessing_util import put_queue_none, clear_queue

from src.view.show_cam import show_cam, show_text, show_line
from src.Integration.service_v1.controller.plat_controller import PlatController
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.fetch_api_controller import FetchAPIController
from src.Integration.service_v1.controller.vehicle_history_controller import VehicleHistoryController
from src.controllers.utils.util import (
    convert_bbox_to_decimal,
    print_normalized_points,
    convert_decimal_to_bbox, 
    crop_frame, 
    most_freq, 
    find_closest_strings_dict, 
    check_db, 
    parking_space_vehicle_counter,
    add_overlay,
    draw_points_and_lines,
    draw_tracking_points,
    resize_image
)
from src.controllers.utils.display import draw_box


class DetectionControllerV6:
    def __init__(self, arduino_idx, vehicle_plate_result_queue=None):
        self.clicked_points = []
        self.arduino_idx = arduino_idx
        self.vehicle_plate_result_queue = vehicle_plate_result_queue
        self.stopped = mp.Event()
        self.vehicle_thread = None
        self.floor_id = 0
        self.cam_id = ""
        self._current_frame = None
        self._current_result = None
        self.vehicle_processing_thread = None
        self.results_lock = threading.Lock()

        self.plate_no = ""
        self.centroids = []
        self.width, self.height = 0, 0
        self.car_bboxes = []
        self.poly_points = []
        self.tracking_points = []

        self.db_floor = FloorController()
        self.db_vehicle_history = VehicleHistoryController()

        self._model_built_event = mp.Event()
        self.lock_frame = threading.Lock()
        # self.callback_process_result_func = callback_process_result_func

    def start(self):
        print("[Thread] Starting vehicle detection thread...")
        self.vehicle_thread = threading.Thread(target=self.detect_vehicle_work_thread)
        self.vehicle_thread.start()

        # print("[Thread] Starting result processing thread...")
        # self.vehicle_processing_thread = threading.Thread(target=self.vehicle_process_work_thread)
        # self.vehicle_processing_thread.start()

    # def process_frame(self, frame, floor_id, cam_id):
    #     self._current_frame = frame.copy()
    #     self.floor_id, self.cam_id = floor_id, cam_id
    #     height, width = frame.shape[:2]

    #     self.poly_points, self.tracking_points, frame = crop_frame(
    #         frame=frame, height=height, width=width, 
    #         floor_id=floor_id, cam_id=cam_id
    #     )

    def process_frame(self, frame_bundle: dict):
        self._current_frame = frame_bundle

    def detect_vehicle_work_thread(self):
        vehicle_model = YOLO(config.MODEL_PATH)
        plate_model = YOLO(config.MODEL_PATH_PLAT_v2)
        vehicle_detector = VehicleDetector(vehicle_model, plate_model)
        self._model_built_event.set()

        while True:
            if self.stopped.is_set():
                # print("self.stopped.is_set()")
                break
            
            # print("loop here", time.time())

            # if self._current_frame is None or self._current_frame.size == 0:
            #     print("self._current_frame is None or self._current_frame.size == 0")
            #     continue

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

            # print(frame.shape)
            
            # self.car_detection_result = result dari yolo
            # TODO model detect disimi
            # put cropped car
            # pakai try except
            try:
                # self.floor_id, self.cam_id = floor_id, cam_id
                height, width = frame.shape[:2]

                poly_points, tracking_points, _ = crop_frame(
                    frame=frame, height=height, width=width, 
                    floor_id=floor_id, cam_id=cam_id
                )

                # print("frame.shape", frame.shape)

                # frame_resized = resize_image(frame, 720, 720)
                # print(frame_resized.shape)
                # results = vehicle_model.predict(frame, conf=0.25, device="cuda:0", verbose=False, classes=[2, 7, 5])
                # print(results)
                # print("masukkkkkkk")
                # bboxes = [result.boxes.xyxyn.cpu() for result in results]
                # print(bboxes)

                vehicle_plate_data = vehicle_detector.detect_vehicle(arduino_idx=self.arduino_idx, frame=frame, floor_id=floor_id, cam_id=cam_id, poly_points=poly_points, tracking_points=tracking_points)

                # print("vehicle_plate_data", vehicle_plate_data)

                if vehicle_plate_data is not None and isinstance(vehicle_plate_data, dict):
                    # vehicle_plate_data = {
                    #     "object_id": vehicle_plate_data["object_id"],
                    #     "bg_color": vehicle_plate_data["bg_color"],
                    #     "frame": vehicle_plate_data["frame"],
                    #     "floor_id": vehicle_plate_data["floor_id"],
                    #     "cam_id": vehicle_plate_data["cam_id"],
                    #     "arduino_idx": vehicle_plate_data["arduino_idx"],
                    #     "car_direction": vehicle_plate_data["car_direction"],
                    #     "start_line": vehicle_plate_data["start_line"],
                    #     "end_line": vehicle_plate_data["end_line"]
                    # }
                    bbox = vehicle_plate_data.pop("bbox")

                    # if self.callback_process_result_func is not None:
                    #     self.callback_process_result_func(vehicle_plate_data)

                    if self.vehicle_plate_result_queue is not None:
                        # print("vehicle_plate_data", vehicle_plate_data)
                        print("size: ", self.vehicle_plate_result_queue.qsize())
                        self.vehicle_plate_result_queue.put(vehicle_plate_data)

                # print("self.car_bboxes: ", self.car_bboxes)
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

        if self.vehicle_processing_thread is not None:
            self.vehicle_processing_thread.join()

        # Clear all queues
        clear_queue(self.vehicle_plate_result_queue)

        print("[Controller] All processes and threads stopped.")