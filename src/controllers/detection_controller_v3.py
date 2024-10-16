import os
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
from src.models.vehicle_detection_model_v3 import VehicleDetector
from utils.multiprocessing_util import put_queue_none, clear_queue

from src.view.show_cam import show_cam, show_text, show_line
from src.Integration.service_v1.controller.plat_controller import PlatController
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.fetch_api_controller import FetchAPIController
from src.Integration.service_v1.controller.vehicle_history_controller import VehicleHistoryController
from src.controllers.utils.util import (
    convert_bbox_to_decimal, 
    convert_decimal_to_bbox, 
    crop_frame, 
    most_freq, 
    find_closest_strings_dict, 
    check_db, 
    parking_space_vehicle_counter
)
from src.controllers.utils.display import draw_box


class DetectionControllerV3:
    def __init__(self, arduino_idx):
        self.arduino_idx = arduino_idx
        self.vehicle_result_queue = mp.Queue()
        self.stopped = mp.Event()
        self.vehicle_thread = None
        self.vehicle_bounding_boxes = []
        self.floor_id = 0
        self.cam_id = ""
        self._current_frame = None
        self._current_result = None
        self.vehicle_processing_thread = None
        self.results_lock = threading.Lock()

        self.plate_no = ""
        self.container_plate_no = []
        self.centroids = []
        self.width, self.height = 0, 0
        self.status_register = False
        self.car_direction = None
        self.prev_centroid = None
        self.num_skip_centroid = 0
        self.centroid_sequence = []
        self.car_bboxes = []
        self.poly_points = []
        self.last_result_plate_no = ""

        self.db_floor = FloorController()

    def start(self):
        print("[Thread] Starting vehicle detection thread...")
        self.vehicle_thread = threading.Thread(target=self.detect_vehicle_work_thread)
        self.vehicle_thread.start()

        print("[Thread] Starting result processing thread...")
        self.vehicle_processing_thread = threading.Thread(target=self.vehicle_process_work_thread)
        self.vehicle_processing_thread.start()

    def process_frame(self, frame, floor_id, cam_id):
        self._current_frame = frame.copy()
        self.floor_id = floor_id
        self.cam_id = cam_id

        self.height, self.width = frame.shape[:2]

        slot = self.db_floor.get_slot_by_id(floor_id)
        total_slot = slot["slot"]
        vehicle_total = slot["vehicle_total"]

        self.poly_points, frame = crop_frame(frame=frame, height=self.height, width=self.width, floor_id=floor_id, cam_id=cam_id)

        show_text(f"Floor : {floor_id} {cam_id}", frame, 5, 50)
        show_text(f"Plate No. : {self.last_result_plate_no}", frame, 5, 100)
        show_text(f"P. Spaces Available : {total_slot}", frame, 5, 150, (0, 255, 0) if total_slot > 0 else (0, 0, 255))
        show_text(f"Car Total : {vehicle_total}", frame, 5, 200)
        show_line(frame, self.poly_points[0], self.poly_points[1])
        show_line(frame, self.poly_points[2], self.poly_points[3])

        draw_box(frame=frame, boxes=self.car_bboxes)
        window_name = f"FLOOR {floor_id}: {cam_id}"
        show_cam(window_name, frame)

        cv2.setMouseCallback(window_name, self.mouse_event) 

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Koordinat yang diklik: ({x}, {y})") 
    #         print(convert_bbox_to_decimal((self.height, self.width), [[[x, y]]]))

    def get_vehicle_results(self):
        return self._current_result
    
    def vehicle_process_work_thread(self):
        while True:
            if self.stopped.is_set():
                break

            try:
                result = self.vehicle_result_queue.get()

                if result is None:
                    print("vehicle_result_queue is None", result)
                    continue

                self._current_result = result

            except Exception as e:
                print(f"Error in vehicle_result_queue work thread: {e}")


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
            # try:
            if frame is None or frame.size == 0:
                print("Empty or invalid frame received.")
                return None

            _, self.car_bboxes = vehicle_detector.detect_vehicle(arduino_idx=self.arduino_idx, frame=frame, floor_id=self.floor_id, cam_id=self.cam_id, poly_points=self.poly_points)

            # print("self.car_bboxes: ", self.car_bboxes)
            # except Exception as e:
            #     print(f"Error in vehicle_detector: {e}")

    def stop(self):
        print("[Controller] Stopping detection processes and threads...")
        self.stopped.set()

        put_queue_none(self.vehicle_result_queue)

        # Stop threads
        if self.vehicle_thread is not None:
            self.vehicle_thread.join()
            self.vehicle_thread = None

        if self.vehicle_processing_thread is not None:
            self.vehicle_processing_thread.join()

        # Clear all queues
        clear_queue(self.vehicle_result_queue)

        print("[Controller] All processes and threads stopped.")