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
from src.models.vehicle_detection_model_v5 import VehicleDetector
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
    draw_tracking_points
)
from src.controllers.utils.display import draw_box


class DetectionControllerV5:
    def __init__(self, arduino_idx):
        self.clicked_points = []
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
        self.centroids = []
        self.width, self.height = 0, 0
        self.status_register = False
        self.car_direction = None
        self.prev_centroid = None
        self.num_skip_centroid = 0
        self.centroid_sequence = []
        self.car_bboxes = []
        self.poly_points = []
        self.tracking_points = []
        self.last_result_plate_no = ""

        self.db_floor = FloorController()
        self.db_vehicle_history = VehicleHistoryController()

    def start(self):
        print("[Thread] Starting vehicle detection thread...")
        self.vehicle_thread = threading.Thread(target=self.detect_vehicle_work_thread)
        self.vehicle_thread.start()

        print("[Thread] Starting result processing thread...")
        self.vehicle_processing_thread = threading.Thread(target=self.vehicle_process_work_thread)
        self.vehicle_processing_thread.start()

    def process_frame(self, frame, floor_id, cam_id, is_debug=False):
        self._current_frame = frame.copy()
        self.floor_id, self.cam_id = floor_id, cam_id
        height, width = frame.shape[:2]

        slot = self.db_floor.get_slot_by_id(floor_id)
        total_slot, vehicle_total = slot["slot"], slot["vehicle_total"]

        self.poly_points, self.tracking_points, frame = crop_frame(
            frame=frame, height=height, width=width, 
            floor_id=floor_id, cam_id=cam_id
        )

        draw_tracking_points(frame, self.tracking_points, (height, width))

        last_plate_no = self.db_vehicle_history.get_vehicle_history_by_floor_id(floor_id)["plate_no"]
        plate_no = last_plate_no if last_plate_no else ""

        add_overlay(frame, floor_id, cam_id, self.poly_points, plate_no, total_slot, vehicle_total)

        if is_debug:
            draw_points_and_lines(frame)
            draw_box(frame=frame, boxes=self.car_bboxes)
        else:
            draw_box(frame=frame, boxes=self.car_bboxes)

        window_name = f"FLOOR {floor_id}: {cam_id}"
        show_cam(window_name, frame)
        cv2.setMouseCallback(
            window_name, 
            self._mouse_event_debug if is_debug else self._mouse_event, 
            param=frame
        )

    def _mouse_event_debug(self, event, x, y, flags, frame):
        """Handle mouse events in debug mode."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
            print(f"Clicked coordinates: ({x}, {y})")

            normalized_points = convert_bbox_to_decimal((frame.shape[:2]), [self.clicked_points])
            print_normalized_points(normalized_points)

            draw_points_and_lines(frame)
            show_cam(f"FLOOR {self.floor_id}: {self.cam_id}", frame)

    def _mouse_event(self, event, x, y, flags, frame):
        """Handle mouse events for normal mode."""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked coordinates: ({x}, {y})")
            print(convert_bbox_to_decimal((frame.shape[:2]), [[[x, y]]]))

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

            _, self.car_bboxes = vehicle_detector.detect_vehicle(arduino_idx=self.arduino_idx, frame=frame, floor_id=self.floor_id, cam_id=self.cam_id, poly_points=self.poly_points, tracking_points=self.tracking_points)

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