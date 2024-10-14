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
from src.controller.matrix_controller import MatrixController
from src.models.vehicle_detection_model import VehicleDetector
from src.models.plate_detection_model import plate_detection_process
from src.models.image_restoration_model import image_restoration
from src.models.text_detection_model import text_detection
from src.models.character_recognition_model import character_recognition
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


class DetectionController:
    def __init__(self, arduino_idx, matrix_total):
        self.arduino_idx = arduino_idx
        self.matrix_text = MatrixController(arduino_idx, 0, 100)
        self.matrix_text.start()
        self.matrix = matrix_total
        self.matrix.start(self.matrix.get_total())
        self.vehicle_result_queue = mp.Queue()
        self.plate_result_queue = mp.Queue()
        self.img_restoration_result_queue = mp.Queue()
        self.text_detection_result_queue = mp.Queue()
        self.stopped = mp.Event()
        self.char_recognize_result_queue = mp.Queue()
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

        # print("[Thread] Starting result processing thread...")
        # self.result_processing_thread = threading.Thread(target=self.post_process_work_thread)
        # self.result_processing_thread.start()

        print("[Process] Starting image_restoration, text_detection, character_recognition process...")
        self.img_restore_text_char_process = mp.Process(target=image_restoration, args=(self.stopped, self.plate_result_queue, self.char_recognize_result_queue))
        self.img_restore_text_char_process.start()

    def get_plate_number(self):
        if self._current_result and 'plate_no' in self._current_result:
            return self._current_result['plate_no']
        return None

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

    def get_results(self):
        return self._current_result

    def post_process_work_thread(self):
        while True:
            if self.stopped.is_set():
                break

            try:
                result = self.char_recognize_result_queue.get()

                if result is None:
                    print("Result is None", result)
                    continue

                self._current_result = result

                floor_id = result.get("floor_id", 0)
                cam_id = result.get("cam_id", "")
                car_direction = result.get("car_direction", None)  # True or False
                mobil_masuk = result.get("mobil_masuk", None)  # True or False
                arduino_idx = result.get("arduino_idx", None)
                start_line = result.get("start_line", None)
                end_line = result.get("end_line", None)  # Example: (start=True, end=False)
                plate_no = result.get("plate_no", None)  # Use None to check for absence

                # print(f'start_line: {start_line} & end_line: {end_line} & plate_no: {plate_no} & car_direction: {car_direction}')

                # Append plate_no if start_line and end_line are both True
                if start_line and end_line and plate_no is not None:  # Check that plate_no is not None
                    self.container_plate_no.append(plate_no)
                    print(f'plate_no: {plate_no}')

                # If both start_line and end_line are False, process the collected plate numbers
                if not start_line and not end_line:
                    if len(self.container_plate_no) > 0:
                        print(f'self.container_plate_no: {self.container_plate_no}')
                        plate_no_max = most_freq(self.container_plate_no)
                        plate_no_detected = plate_no_max
                        status_plate_no = check_db(plate_no_detected)

                        plate_no_is_registered = True
                        if not status_plate_no:
                            logger.write(
                                f"Warning, plat is unregistered, reading container text!! : {plate_no_detected}",
                                logger.WARN
                            )
                            plate_no_is_registered = False

                        current_max_slot, current_slot_update, current_vehicle_total_update = parking_space_vehicle_counter(floor_id=floor_id, cam_id=cam_id, arduino_idx=arduino_idx, car_direction=car_direction, plate_no=plate_no_detected)

                        matrix_update = MatrixController(arduino_idx, max_car=current_max_slot, total_car=current_slot_update)
                        available_space = matrix_update.get_total()
                        self.total_slot = current_max_slot - available_space
                        self.last_result_plate_no = plate_no_detected

                        print(f"PLAT_NO : {plate_no_detected}, AVAILABLE PARKING SPACES : {available_space}, STATUS : {'TAMBAH' if not car_direction else 'KURANG'}, VEHICLE_TOTAL: {current_vehicle_total_update}, FLOOR : {floor_id}, CAMERA : {cam_id}, TOTAL_FRAME: {len(self.container_plate_no)}")

                        self.db_vehicle_history.create_vehicle_history_record(plate_no=self.last_result_plate_no, floor_id=floor_id, camera=cam_id)
                        
                        # self.send_plate_data(floor_id=current_floor_position, plate_no=plate_no, cam_position=current_cam_position)

                        # print('=' * 30 + " BORDER: LAST RESULT " + '=' * 30)

                        char = "H" if plate_no_is_registered else "M"
                        matrix_text = f"{plate_no_detected},{char};"
                        self.matrix_text.write_arduino(matrix_text)
                        self.container_plate_no = []

                        if not self.db_plate.check_exist_plat(plate_no_detected):
                            plate_no_is_registered = False
                            logger.write(
                                f"WARNING THERE IS NO PLAT IN DATABASE!!! text: {plate_no_detected}, status: {car_direction}",
                                logger.WARNING
                            )

            except Exception as e:
                print(f"Error in post-process work thread: {e}")

    def detect_vehicle_plate_work_thread(self):
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

            _, self.car_bboxes = vehicle_detector.detect_vehicle(arduino_idx=self.arduino_idx, frame=frame, floor_id=self.floor_id, cam_id=self.cam_id, matrix=self.matrix, poly_points=self.poly_points)

            # print("self.car_bboxes: ", self.car_bboxes)
            # except Exception as e:
            #     print(f"Error in vehicle_detector: {e}")

    def stop(self):
        print("[Controller] Stopping detection processes and threads...")
        self.stopped.set()

        put_queue_none(self.vehicle_result_queue)
        put_queue_none(self.plate_result_queue)
        put_queue_none(self.img_restoration_result_queue)
        put_queue_none(self.text_detection_result_queue)
        put_queue_none(self.char_recognize_result_queue)

        # Stop threads
        if self.vehicle_plate_thread is not None:
            self.vehicle_plate_thread.join()
            self.vehicle_plate_thread  = None

        # if self.result_processing_thread is not None:
        #     self.result_processing_thread.join()

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


    # def post_process_work_thread(self):
    #     while True:
    #         if self.stopped.is_set():
    #             break

    #         try:
    #             result = self.char_recognize_result_queue.get()

    #             if result is None:
    #                 print("result is None", result)
    #                 continue

    #             self._current_result = result

    #         except Exception as e:
    #             print(f"Error in get plate_no: {e}")