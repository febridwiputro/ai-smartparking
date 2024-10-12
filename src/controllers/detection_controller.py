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
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.fetch_api_controller import FetchAPIController
from src.Integration.service_v1.controller.vehicle_history_controller import VehicleHistoryController
from src.utils import get_centroids
from src.controllers.utils.util import convert_bbox_to_decimal, convert_decimal_to_bbox, crop_frame
from src.controllers.utils.display import draw_box


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
        self.floor_id = 0
        self.cam_id = ""
        self._current_frame = None
        self._current_result = None
        self.result_processing_thread = None
        self.results_lock = threading.Lock()

        self.passed_a = 0
        self.plate_no = ""
        self.container_plate_no = []
        self.mobil_masuk = False
        self.track_id = 0
        self.centroids = []
        self.passed: int = 0
        self.width, self.height = 0, 0
        self.status_register = False
        self.car_direction = None
        self.prev_centroid = None
        self.num_skip_centroid = 0
        self.centroid_sequence = []
        self.db_floor = FloorController()
        self.db_mysn = FetchAPIController()
        self.db_vehicle_history = VehicleHistoryController()
        self.car_bboxes = []
        self.poly_points = []

    def start(self):
        print("[Thread] Starting vehicle detection thread...")
        self.vehicle_thread = threading.Thread(target=self.detect_vehicle_work_thread)
        self.vehicle_thread.start()

        # Start result processing thread
        self.result_processing_thread = threading.Thread(target=self.post_process_work_thread)
        self.result_processing_thread.start()

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

    def get_plate_number(self):
        if self._current_result and 'plate_no' in self._current_result:
            return self._current_result['plate_no']
        return None

    def process_frame(self, frame, floor_id, cam_id):
        self.height, self.width = frame.shape[:2]

        slot = self.db_floor.get_slot_by_id(floor_id)
        total_slot = slot["slot"]
        vehicle_total = slot["vehicle_total"]

        self.poly_points, frame = crop_frame(frame=frame, height=self.height, width=self.width, floor_id=floor_id, cam_id=cam_id)

        show_text(f"Floor : {floor_id} {cam_id}", frame, 5, 50)
        plate_no = self.get_plate_number()
        show_text(f"Plate No. : {plate_no}", frame, 5, 100)
        show_text(f"Parking Lot Available : {total_slot}", frame, 5, 150, (0, 255, 0) if total_slot > 0 else (0, 0, 255))
        show_text(f"Car Total : {vehicle_total}", frame, 5, 200)
        show_line(frame, self.poly_points[0], self.poly_points[1])
        show_line(frame, self.poly_points[2], self.poly_points[3])

        draw_box(frame=frame, boxes=self.car_bboxes)
        window_name = f"FLOOR {floor_id}: {cam_id}"
        show_cam(window_name, frame)

        cv2.setMouseCallback(window_name, self.mouse_event) 

        self._current_frame = frame
        self.floor_id = floor_id
        self.cam_id = cam_id

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
                    print("result is None", result)
                    continue

                self._current_result = result

            except Exception as e:
                print(f"Error in get plate_no: {e}")

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

            _, self.car_bboxes = vehicle_detector.detect_vehicle(frame=frame, floor_id=self.floor_id, cam_id=self.cam_id, matrix=self.matrix, poly_points=self.poly_points)

            # except Exception as e:
            #     print(f"Error in vehicle_detector: {e}")
    

    # def post_process_work_thread(self):
    #     while True:
    #         result = self.char_recognize_result_queue.get()
    #         # print("char_recognize_result_queue: ", result)
    #         # process result disini
    #         self._current_result = result

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

        if self.result_processing_thread is not None:
            self.result_processing_thread.join()

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



    def update_slot_and_vehicle_total(self, floor_id, new_slot, new_vehicle_total):
        """Helper function to update the parking slot and vehicle total in the database."""
        self.db_floor.update_slot_by_id(id=floor_id, new_slot=new_slot)
        self.db_floor.update_vehicle_total_by_id(id=floor_id, new_vehicle_total=new_vehicle_total)

    def parking_space_vehicle_counter(self, cam_idx, arduino_idx, is_vehicle_entering, plate_no):
        current_floor_position, current_cam_position = self.check_floor(cam_idx=cam_idx)
        current_data = self.db_floor.get_slot_by_id(current_floor_position)
        current_slot = current_data["slot"]
        current_max_slot = current_data["max_slot"]
        current_vehicle_total = current_data["vehicle_total"]

        # Previous and next floor positions
        prev_floor_position = current_floor_position - 1
        next_floor_position = current_floor_position - 1

        prev_data = self.db_floor.get_slot_by_id(prev_floor_position)
        next_data = self.db_floor.get_slot_by_id(next_floor_position)

        # Vehicle history lookup
        get_plate_history = self.db_vehicle_history.get_vehicle_history_by_plate_no(plate_no=plate_no)
        print("get_plate_history: ", get_plate_history)

        if is_vehicle_entering:  # Vehicle IN
            # if get_plate_history:
            #     if get_plate_history[0]['floor_id'] != current_floor_position:
            #         print(f"Update vehicle history karena floor_id tidak sesuai: {get_plate_history[0]['floor_id']} != {current_floor_position}")
                    
            #         # Update vehicle history
            #         update_plate_history = self.db_vehicle_history.update_vehicle_history_by_plate_no(
            #             plate_no=plate_no, 
            #             floor_id=current_floor_position, 
            #             camera=current_cam_position
            #         )

            #         if update_plate_history:
            #             print(f"Vehicle history updated for plate_no: {plate_no} to floor_id: {current_floor_position}")
            #         else:
            #             print(f"Failed to update vehicle history for plate_no: {plate_no}")

            # if get_plate_history:
            #     if get_plate_history[0]['floor_id'] != current_floor_position:
            #         print(f"Update vehicle history karena floor_id tidak sesuai: {get_plate_history[0]['floor_id']} != {current_floor_position}")
                    
            #         # Update vehicle history
            #         update_plate_history = self.db_vehicle_history.update_vehicle_history_by_plate_no(
            #             plate_no=plate_no, 
            #             floor_id=current_floor_position, 
            #             camera=current_cam_position
            #         )

            #         if update_plate_history:
            #             print(f"Vehicle history updated for plate_no: {plate_no} to floor_id: {current_floor_position}")
            #         else:
            #             print(f"Failed to update vehicle history for plate_no: {plate_no}")

            #     if current_floor_position == 5 and get_plate_history[0]['floor_id'] == 4:
            #         current_slot_update = current_slot + 1
            #         self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

            #         current_vehicle_total_update = current_vehicle_total - 1
            #         self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)
            #         print(f"Updated current_slot to {current_slot_update} and vehicle_total to {current_vehicle_total_update}")

            #     elif current_floor_position == 4 and get_plate_history[0]['floor_id'] == 3:
            #         current_slot_update = current_slot + 1
            #         self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

            #         current_vehicle_total_update = current_vehicle_total - 1
            #         self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)
            #         print(f"Updated current_slot to {current_slot_update} and vehicle_total to {current_vehicle_total_update}")

            #     elif current_floor_position == 3 and get_plate_history[0]['floor_id'] == 2:
            #         current_slot_update = current_slot + 1
            #         self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

            #         current_vehicle_total_update = current_vehicle_total - 1
            #         self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)
            #         print(f"Updated current_slot to {current_slot_update} and vehicle_total to {current_vehicle_total_update}")

            print("VEHICLE - IN")
            print(f'CURRENT FLOOR : {current_floor_position} && PREV FLOOR {prev_floor_position}')  

            # Handling the case of an empty current slot
            if current_slot == 0:
                print("Current slot is empty, updating counts...")
                current_slot_update = current_slot  # remains 0
                current_vehicle_total_update = current_vehicle_total + 1
                self.update_slot_and_vehicle_total(current_floor_position, current_slot_update, current_vehicle_total_update)

                # Update previous floor
                if prev_floor_position > 0:
                    prev_slot_update = prev_data["slot"] + 1 if prev_data["slot"] < prev_data["max_slot"] else prev_data["slot"]
                    prev_vehicle_total_update = prev_data["vehicle_total"] - 1 if prev_data["vehicle_total"] > 0 else prev_data["vehicle_total"]
                    self.update_slot_and_vehicle_total(prev_floor_position, prev_slot_update, prev_vehicle_total_update)

            elif current_slot > 0 and current_slot < current_max_slot:
                current_slot_update = current_slot - 1
                current_vehicle_total_update = current_vehicle_total + 1
                self.update_slot_and_vehicle_total(current_floor_position, current_slot_update, current_vehicle_total_update)

                if prev_floor_position > 0:
                    prev_slot_update = prev_data["slot"] + 1
                    prev_vehicle_total_update = prev_data["vehicle_total"] - 1
                    self.update_slot_and_vehicle_total(prev_floor_position, prev_slot_update, prev_vehicle_total_update)

        else:  # Vehicle OUT
            print("VEHICLE - OUT")
            print(f'CURRENT FLOOR : {current_floor_position} && NEXT FLOOR {next_floor_position}')            
            if current_slot == 0:
                if current_vehicle_total > 0:
                    current_slot_update = current_slot + 1
                    current_vehicle_total_update = current_vehicle_total - 1
                    self.update_slot_and_vehicle_total(current_floor_position, current_slot_update, current_vehicle_total_update)

                    if next_floor_position <= self.max_floors:
                        next_slot = next_data["slot"]
                        if next_slot < next_data["max_slot"]:
                            next_slot_update = next_slot - 1
                            next_vehicle_total_update = next_data["vehicle_total"] + 1
                            self.update_slot_and_vehicle_total(next_floor_position, next_slot_update, next_vehicle_total_update)

                # Handling overcapacity scenarios...
                elif current_vehicle_total > current_max_slot:
                    current_slot_update = current_slot
                    current_vehicle_total_update = current_vehicle_total + 1
                    self.update_slot_and_vehicle_total(current_floor_position, current_slot_update, current_vehicle_total_update)

                    if next_floor_position <= self.max_floors:
                        next_slot = next_data["slot"]
                        if next_slot < next_data["max_slot"]:
                            next_slot_update = next_slot - 1
                            next_vehicle_total_update = next_data["vehicle_total"] + 1
                            self.update_slot_and_vehicle_total(next_floor_position, next_slot_update, next_vehicle_total_update)

            elif 0 < current_slot <= current_max_slot:
                current_slot_update = current_slot + 1
                self.update_slot_and_vehicle_total(current_floor_position, current_slot_update, current_vehicle_total - 1)

                if next_floor_position <= self.max_floors:
                    next_slot = next_data["slot"]
                    if next_slot < next_data["max_slot"]:
                        next_slot_update = next_slot - 1
                        next_vehicle_total_update = next_data["vehicle_total"] + 1
                        self.update_slot_and_vehicle_total(next_floor_position, next_slot_update, next_vehicle_total_update)

        print(f"Updated Slot for Floor {current_floor_position}: {current_slot_update}")

        matrix_update = MatrixController(arduino_idx, max_car=current_max_slot, total_car=current_slot_update)
        available_space = matrix_update.get_total()
        self.total_slot = current_max_slot - available_space
        self.plate_no = plate_no

        print(f"PLAT_NO : {plate_no}, AVAILABLE PARKING SPACES : {available_space}, STATUS : {'TAMBAH' if not is_vehicle_entering else 'KURANG'}, VEHICLE_TOTAL: {current_vehicle_total_update}, FLOOR : {current_floor_position}, CAMERA : {current_cam_position}, TOTAL_FRAME: {len(self.container_plate_no)}")

        self.db_vehicle_history.create_vehicle_history_record(plate_no=plate_no, floor_id=current_floor_position, camera=current_cam_position)
        
        # self.send_plate_data(floor_id=current_floor_position, plate_no=plate_no, cam_position=current_cam_position)

        print('=' * 30 + " LINE BORDER " + '=' * 30)



    # def parking_space_vehicle_counter(self, cam_idx, arduino_idx, status_car, plate_no):
    #     current_floor_position, current_cam_position = self.check_floor(cam_idx=cam_idx)
    #     current_data = self.db_floor.get_slot_by_id(current_floor_position)
    #     current_slot = current_data["slot"]
    #     current_max_slot = current_data["max_slot"]
    #     current_vehicle_total = current_data["vehicle_total"]
    #     current_slot_update = current_slot
    #     current_vehicle_total_update = current_vehicle_total

    #     prev_floor_position = current_floor_position - 1
    #     prev_data = self.db_floor.get_slot_by_id(prev_floor_position)
    #     prev_slot = prev_data["slot"]
    #     prev_max_slot = prev_data["max_slot"]
    #     prev_vehicle_total = prev_data["vehicle_total"]
    #     prev_slot_update = prev_slot
    #     prev_vehicle_total_update = prev_vehicle_total

    #     next_floor_position = current_floor_position - 1
    #     next_data = self.db_floor.get_slot_by_id(next_floor_position)
    #     next_slot = next_data["slot"]
    #     next_max_slot = next_data["max_slot"]
    #     next_vehicle_total = next_data["vehicle_total"]
    #     next_slot_update = next_slot
    #     next_vehicle_total_update = next_vehicle_total

    #     get_plate_history = self.db_vehicle_history.get_vehicle_history_by_plate_no(plate_no=plate_no)
    #     print("get_plate_history: ", get_plate_history)

    #     # NAIK / MASUK
    #     if not status_car:
    #         # if get_plate_history:
    #         #     if get_plate_history[0]['floor_id'] != current_floor_position:
    #         #         print(f"Update vehicle history karena floor_id tidak sesuai: {get_plate_history[0]['floor_id']} != {current_floor_position}")
                    
    #         #         # Update vehicle history
    #         #         update_plate_history = self.db_vehicle_history.update_vehicle_history_by_plate_no(
    #         #             plate_no=plate_no, 
    #         #             floor_id=current_floor_position, 
    #         #             camera=current_cam_position
    #         #         )

    #         #         if update_plate_history:
    #         #             print(f"Vehicle history updated for plate_no: {plate_no} to floor_id: {current_floor_position}")
    #         #         else:
    #         #             print(f"Failed to update vehicle history for plate_no: {plate_no}")

    #         # if get_plate_history:
    #         #     if get_plate_history[0]['floor_id'] != current_floor_position:
    #         #         print(f"Update vehicle history karena floor_id tidak sesuai: {get_plate_history[0]['floor_id']} != {current_floor_position}")
                    
    #         #         # Update vehicle history
    #         #         update_plate_history = self.db_vehicle_history.update_vehicle_history_by_plate_no(
    #         #             plate_no=plate_no, 
    #         #             floor_id=current_floor_position, 
    #         #             camera=current_cam_position
    #         #         )

    #         #         if update_plate_history:
    #         #             print(f"Vehicle history updated for plate_no: {plate_no} to floor_id: {current_floor_position}")
    #         #         else:
    #         #             print(f"Failed to update vehicle history for plate_no: {plate_no}")

    #         #     if current_floor_position == 5 and get_plate_history[0]['floor_id'] == 4:
    #         #         current_slot_update = current_slot + 1
    #         #         self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

    #         #         current_vehicle_total_update = current_vehicle_total - 1
    #         #         self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)
    #         #         print(f"Updated current_slot to {current_slot_update} and vehicle_total to {current_vehicle_total_update}")

    #         #     elif current_floor_position == 4 and get_plate_history[0]['floor_id'] == 3:
    #         #         current_slot_update = current_slot + 1
    #         #         self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

    #         #         current_vehicle_total_update = current_vehicle_total - 1
    #         #         self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)
    #         #         print(f"Updated current_slot to {current_slot_update} and vehicle_total to {current_vehicle_total_update}")

    #         #     elif current_floor_position == 3 and get_plate_history[0]['floor_id'] == 2:
    #         #         current_slot_update = current_slot + 1
    #         #         self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

    #         #         current_vehicle_total_update = current_vehicle_total - 1
    #         #         self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)
    #         #         print(f"Updated current_slot to {current_slot_update} and vehicle_total to {current_vehicle_total_update}")

    #         print("VEHICLE - IN")
    #         print(f'CURRENT FLOOR : {current_floor_position} && PREV FLOOR {prev_floor_position}')  

    #         if current_slot == 0:
    #             print("UPDATE 0")
    #             current_slot_update = current_slot
    #             self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

    #             current_vehicle_total_update = current_vehicle_total + 1
    #             self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)

    #             if prev_floor_position > 1:
    #                 if prev_slot == 0:
    #                     if prev_vehicle_total > prev_max_slot:
    #                         prev_slot_update = prev_slot
    #                         self.db_floor.update_slot_by_id(id=prev_floor_position, new_slot=prev_slot_update)

    #                         prev_vehicle_total_update = prev_vehicle_total - 1
    #                         self.db_floor.update_vehicle_total_by_id(id=prev_floor_position, new_vehicle_total=prev_vehicle_total_update)
    #                     else:
    #                         prev_slot_update = prev_slot + 1
    #                         self.db_floor.update_slot_by_id(id=prev_floor_position, new_slot=prev_slot_update)

    #                         prev_vehicle_total_update = prev_vehicle_total - 1
    #                         self.db_floor.update_vehicle_total_by_id(id=prev_floor_position, new_vehicle_total=prev_vehicle_total_update)                            

    #                 elif prev_slot > 0 and prev_slot < prev_max_slot:
    #                     prev_slot_update = prev_slot + 1
    #                     self.db_floor.update_slot_by_id(id=prev_floor_position, new_slot=prev_slot_update)

    #                     prev_vehicle_total_update = prev_vehicle_total - 1
    #                     self.db_floor.update_vehicle_total_by_id(id=prev_floor_position, new_vehicle_total=prev_vehicle_total_update)

    #         elif current_slot > 0 and current_slot <= current_max_slot:
    #             current_slot_update = current_slot - 1
    #             print("current_slot_update: ", current_slot_update)
    #             self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

    #             current_vehicle_total_update = current_vehicle_total + 1
    #             print("current_vehicle_total_update: ", current_vehicle_total_update)
    #             self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)

    #             if prev_floor_position > 1:
    #                 print("IN 1")
    #                 if prev_slot == 0:
    #                     if prev_vehicle_total > prev_max_slot:
    #                         prev_slot_update = prev_slot
    #                         self.db_floor.update_slot_by_id(id=prev_floor_position, new_slot=prev_slot_update)

    #                         prev_vehicle_total_update = prev_vehicle_total - 1
    #                         self.db_floor.update_vehicle_total_by_id(id=prev_floor_position, new_vehicle_total=prev_vehicle_total_update)
    #                     else:
    #                         prev_slot_update = prev_slot + 1
    #                         self.db_floor.update_slot_by_id(id=prev_floor_position, new_slot=prev_slot_update)

    #                         prev_vehicle_total_update = prev_vehicle_total - 1
    #                         self.db_floor.update_vehicle_total_by_id(id=prev_floor_position, new_vehicle_total=prev_vehicle_total_update)                            

    #                 elif prev_slot > 0 and prev_slot < prev_max_slot:
    #                     print("IN 2")
    #                     prev_slot_update = prev_slot + 1
    #                     print("prev_slot_update: ", prev_slot_update)
    #                     print("prev_slot_update: ", prev_slot_update)

    #                     self.db_floor.update_slot_by_id(id=prev_floor_position, new_slot=prev_slot_update)

    #                     prev_vehicle_total_update = prev_vehicle_total - 1
    #                     print("prev_vehicle_total_update: ", prev_vehicle_total_update)
    #                     self.db_floor.update_vehicle_total_by_id(id=prev_floor_position, new_vehicle_total=prev_vehicle_total_update)

    #     # TURUN / KELUAR
    #     else:
    #         print("VEHICLE - OUT")
    #         print(f'CURRENT FLOOR : {current_floor_position} && NEXT FLOOR {next_floor_position}')            
    #         if current_slot == 0:
    #             if current_vehicle_total > 0 and current_vehicle_total <= current_max_slot:
    #                 print("CURRENT OUT 1")
    #                 current_slot_update = current_slot + 1
    #                 self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

    #                 current_vehicle_total_update = current_vehicle_total - 1
    #                 self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)

    #                 if next_floor_position > 1:
    #                     if next_slot == 0:
    #                         print("NEXT OUT 1")
    #                         if next_vehicle_total >= next_max_slot:
    #                             next_vehicle_total_update = next_vehicle_total_update + 1
    #                             self.db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)
    #                     elif next_slot > 0 and next_slot <= next_max_slot:
    #                         print("NEXT OUT 2")
    #                         next_slot_update = next_slot - 1
    #                         self.db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

    #                         next_vehicle_total_update = next_vehicle_total_update + 1
    #                         self.db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)

    #             elif current_vehicle_total > current_max_slot:
    #                 print("CURRENT OUT 2")
    #                 current_slot_update = current_slot
    #                 self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

    #                 current_vehicle_total_update = current_vehicle_total + 1
    #                 self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)

    #                 if next_floor_position > 1:
    #                     if next_slot == 0:
    #                         if next_vehicle_total > next_max_slot:
    #                             next_vehicle_total_update = next_vehicle_total_update + 1
    #                             self.db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)
    #                     elif next_slot > 0 and next_slot <= next_max_slot:
    #                         next_slot_update = next_slot - 1
    #                         self.db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

    #                         next_vehicle_total_update = next_vehicle_total_update + 1
    #                         self.db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)


    #         elif current_slot > 0 and current_slot <= current_max_slot:
    #             if current_slot == 18:
    #                 print("CURRENT OUT 3")
    #                 current_slot_update = current_slot
    #                 self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)                    
    #             else:
    #                 print("CURRENT OUT 4")
    #                 current_slot_update = current_slot + 1
    #                 self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

    #             if current_vehicle_total == 0:
    #                 current_vehicle_total_update = current_vehicle_total
    #                 self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)
    #             else:
    #                 current_vehicle_total_update = current_vehicle_total - 1
    #                 self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)

    #             if next_floor_position > 1:
    #                 if next_slot == 0:
    #                     print("NEXT OUT 3")
    #                     if next_vehicle_total > next_max_slot:
    #                         next_slot_update = next_slot
    #                         self.db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

    #                         next_vehicle_total_update = next_vehicle_total + 1
    #                         self.db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)
    #                 elif next_slot > 0 and next_slot <= next_max_slot:
    #                     print("NEXT OUT 4")
    #                     next_slot_update = next_slot - 1
    #                     self.db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

    #                     next_vehicle_total_update = next_vehicle_total + 1
    #                     self.db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)
    #                 elif next_slot > next_max_slot:
    #                     print("NEXT OUT 5")
    #                     next_slot_update = next_slot
    #                     self.db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

    #                     next_vehicle_total_update = next_vehicle_total + 1
    #                     self.db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)

    #         print("current_slot_update: ", current_slot_update)
    #         print("next_vehicle_total_update: ", next_vehicle_total_update)

    #     matrix_update = MatrixController(arduino_idx, max_car=current_max_slot, total_car=current_slot_update)
    #     available_space = matrix_update.get_total()
    #     self.total_slot = current_max_slot - available_space
    #     self.plate_no = plate_no

    #     print(f"PLAT_NO : {plate_no}, AVAILABLE PARKING SPACES : {available_space}, STATUS : {'TAMBAH' if not status_car else 'KURANG'}, VEHICLE_TOTAL: {current_vehicle_total_update}, FLOOR : {current_floor_position}, CAMERA : {current_cam_position}, TOTAL_FRAME: {len(self.container_plate_no)}")

    #     self.db_vehicle_history.create_vehicle_history_record(plate_no=plate_no, floor_id=current_floor_position, camera=current_cam_position)
        
    #     # self.send_plate_data(floor_id=current_floor_position, plate_no=plate_no, cam_position=current_cam_position)

    #     print('=' * 30 + " LINE BORDER " + '=' * 30)
