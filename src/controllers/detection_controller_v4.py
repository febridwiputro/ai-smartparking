import os
import threading
import multiprocessing as mp
import cv2
import numpy as np
from ultralytics import YOLO

from src.config.config import config
from src.config.logger import logger
from src.controllers.matrix_controller import MatrixController
from src.models.vehicle_plate_model_v4 import VehicleDetector
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


class DetectionControllerV4:
    def __init__(self, arduino_idx):
        self.arduino_idx = arduino_idx
        self.vehicle_plate_result_queue = mp.Queue()
        self.plate_result_queue = mp.Queue()
        self.img_restoration_result_queue = mp.Queue()
        self.text_detection_result_queue = mp.Queue()
        self.stopped = mp.Event()
        self.img_restore_text_char_queue = mp.Queue()
        self.vehicle_plate_thread = None
        self.img_restore_text_char_process = None
        self.vehicle_bounding_boxes = []
        self.floor_id = 0
        self.cam_id = ""
        self._current_frame = None
        self._current_result = None
        # self.result_processing_thread = None
        self.results_lock = threading.Lock()

        self.passed_a = 0
        self.plate_no = ""
        self.container_plate_no = []
        self.mobil_masuk = False
        self.track_id = 0
        self.centroids = []
        self.width, self.height = 0, 0
        self.status_register = False
        self.car_direction = None
        self.prev_centroid = None
        self.num_skip_centroid = 0
        self.centroid_sequence = []
        self.db_plate = PlatController()
        self.db_floor = FloorController()
        self.db_mysn = FetchAPIController()
        self.db_vehicle_history = VehicleHistoryController()
        self.car_bboxes = []
        self.poly_points = []
        self.last_result_plate_no = ""

    def start(self):
        print("[Thread] Starting vehicle detection thread...")
        self.vehicle_plate_thread = threading.Thread(target=self.detect_vehicle_plate_work_thread)
        self.vehicle_plate_thread.start()

        print("[Thread] Starting result processing thread...")
        self.result_processing_thread = threading.Thread(target=self.plate_process_work_thread)
        self.result_processing_thread.start()

    def process_frame(self, frame, floor_id, cam_id):
        last_plate_no = ""
        self._current_frame = frame.copy()
        self.floor_id = floor_id
        self.cam_id = cam_id

        self.height, self.width = frame.shape[:2]

        slot = self.db_floor.get_slot_by_id(floor_id)
        total_slot = slot["slot"]
        vehicle_total = slot["vehicle_total"]

        self.poly_points, tracking_point, frame = crop_frame(frame=frame, height=self.height, width=self.width, floor_id=floor_id, cam_id=cam_id)

        get_plate_floor_id_history = self.db_vehicle_history.get_vehicle_history_by_floor_id(floor_id=floor_id)
        last_plate_no = get_plate_floor_id_history["plate_no"]

        if last_plate_no:
            plate_no = last_plate_no
        else:
            plate_no = ""

        show_text(f"Floor : {floor_id} {cam_id}", frame, 5, 50)
        show_text(f"Plate No. : {plate_no}", frame, 5, 100)
        show_text(f"P. Spaces Available : {total_slot}", frame, 5, 150, (0, 255, 0) if total_slot > 0 else (0, 0, 255))
        show_text(f"Car Total : {vehicle_total}", frame, 5, 200)
        show_line(frame, self.poly_points[0], self.poly_points[1])
        show_line(frame, self.poly_points[2], self.poly_points[3])

        draw_box(frame=frame, boxes=self.car_bboxes)
        window_name = f"FLOOR {floor_id}: {cam_id}"
        show_cam(window_name, frame)

        # cv2.setMouseCallback(window_name, self.mouse_event) 

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Koordinat yang diklik: ({x}, {y})") 
    #         print(convert_bbox_to_decimal((self.height, self.width), [[[x, y]]]))

    def get_plate_results(self):
        return self._current_result

    def plate_process_work_thread(self):
        while True:
            if self.stopped.is_set():
                break

            try:
                result = self.vehicle_plate_result_queue.get()

                if result is None:
                    print("vehicle_result_queue is None", result)
                    continue

                self._current_result = result

            except Exception as e:
                print(f"Error in vehicle_result_queue work thread: {e}")


    def detect_vehicle_plate_work_thread(self):
        # TODO define YOLO MODEL
        vehicle_model = YOLO(config.MODEL_PATH)
        plate_model = YOLO(config.MODEL_PATH_PLAT_v2)
        vehicle_detector = VehicleDetector(vehicle_model, plate_model, self.vehicle_plate_result_queue)

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

        put_queue_none(self.vehicle_plate_result_queue)
        put_queue_none(self.img_restore_text_char_queue)

        # Stop threads
        if self.vehicle_plate_thread is not None:
            self.vehicle_plate_thread.join()
            self.vehicle_plate_thread  = None

        if self.result_processing_thread is not None:
            self.result_processing_thread.join()

        # Stop processes
        if self.img_restore_text_char_process is not None:
            self.img_restore_text_char_process.join()
            self.img_restore_text_char_process = None

        # Clear all queues
        clear_queue(self.vehicle_plate_result_queue)
        clear_queue(self.img_restore_text_char_queue)

        print("[Controller] All processes and threads stopped.")