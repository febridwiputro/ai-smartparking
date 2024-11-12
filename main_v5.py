import os
import cv2
import threading
import multiprocessing as mp
import json
from ultralytics import YOLO

from src.Integration.arduino import Arduino
from src.config.config import config
from src.controllers.matrix_controller import MatrixController
from src.controllers.detection_controller_v5 import DetectionControllerV5
from src.models.cam_model import CameraV1
from src.controllers.utils.util import check_floor
from src.view.show_cam import show_cam, show_text, show_line
from src.Integration.service_v1.controller.plat_controller import PlatController
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.fetch_api_controller import FetchAPIController
from src.Integration.service_v1.controller.vehicle_history_controller import VehicleHistoryController
from utils.multiprocessing_util import put_queue_none, clear_queue
from src.models.plate_detection_model_v5 import plate_detection
from src.models.image_restoration_model_v5 import image_restoration
from src.models.text_detection_model_v5 import text_detection
from src.models.character_recognition_model_v5 import character_recognition, ModelAndLabelLoader
from src.config.logger import logger
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


class Wrapper:
    def __init__(self) -> None:
        self.previous_object_id = None
        self.object_id_count = {}
        self.vehicle_result_queues = [mp.Queue() for _ in range(2)]
        self.plate_result_queues = [mp.Queue() for _ in range(2)]
        self.img_restoration_result_queues = [mp.Queue() for _ in range(2)]
        self.text_detection_result_queues = [mp.Queue() for _ in range(2)]
        self.char_recognize_result_queues = [mp.Queue() for _ in range(2)]
        self.stopped = mp.Event()
        self.processes = []
        self.container_plate_no = []
        self.last_result_plate_no = None
        self.total_slot = 0
        self.vehicle_thread = None
        self.plate_detection_processes = []
        self.image_restoration_processes = []
        self.text_detection_processes = []
        self.char_recognition_processes = []
        self.vehicle_bounding_boxes = []
        self.floor_id = 0
        self.cam_id = ""
        self._current_frame = None
        self._current_result = None
        self.result_processing_thread = None
        self.db_plate = PlatController()
        self.db_floor = FloorController()
        self.db_mysn = FetchAPIController()
        self.db_vehicle_history = VehicleHistoryController()
        
        self.queue_index = 0

    def start(self):
        # print("[Thread] Starting result processing thread...")
        # self.result_processing_thread = threading.Thread(target=self.post_process_work_thread)
        # self.result_processing_thread.start()

        for idx in range(1):
            self.start_detection_processes(idx)

    def start_detection_processes(self, idx):
        print(f"[Process] Starting plate detection process for Queue {idx + 1}...")
        plate_detection_process = mp.Process(
            target=plate_detection,
            args=(self.stopped, self.vehicle_result_queues[idx], self.plate_result_queues[idx])
        )
        self.plate_detection_processes.append(plate_detection_process)
        plate_detection_process.start()

        # print(f"[Process] Starting image restoration process for Queue {idx + 1}...")
        # image_restoration_process = mp.Process(
        #     target=image_restoration,
        #     args=(self.stopped, self.plate_result_queues[idx], self.img_restoration_result_queues[idx])
        # )
        # self.image_restoration_processes.append(image_restoration_process)
        # image_restoration_process.start()

        # print(f"[Process] Starting text detection process for Queue {idx + 1}...")
        # text_detection_process = mp.Process(
        #     target=text_detection,
        #     args=(self.stopped, self.img_restoration_result_queues[idx], self.text_detection_result_queues[idx])
        # )
        # self.text_detection_processes.append(text_detection_process)
        # text_detection_process.start()

        # print(f"[Process] Starting character recognition process for Queue {idx + 1}...")
        # char_recognition_process = mp.Process(
        #     target=character_recognition,
        #     args=(self.stopped, self.text_detection_result_queues[idx], self.char_recognize_result_queues[idx])
        # )
        # self.char_recognition_processes.append(char_recognition_process)
        # char_recognition_process.start()

    def stop(self):
        print("[Controller] Stopping detection processes and threads...")
        self.stopped.set()

        # Put None in all queues to signal termination
        for idx in range(2):
            put_queue_none(self.vehicle_result_queues[idx])
            put_queue_none(self.plate_result_queues[idx])
            put_queue_none(self.img_restoration_result_queues[idx])
            put_queue_none(self.text_detection_result_queues[idx])
            put_queue_none(self.char_recognize_result_queues[idx])

        # Stop the result processing thread
        if self.result_processing_thread is not None:
            self.result_processing_thread.join()

        # Stop all detection processes
        for idx, process in enumerate(self.plate_detection_processes):
            if process.is_alive():
                process.join()
                print(f"[Process] Plate detection process {idx + 1} stopped.")

        for idx, process in enumerate(self.image_restoration_processes):
            if process.is_alive():
                process.join()
                print(f"[Process] Image restoration process {idx + 1} stopped.")

        for idx, process in enumerate(self.text_detection_processes):
            if process.is_alive():
                process.join()
                print(f"[Process] Text detection process {idx + 1} stopped.")

        for idx, process in enumerate(self.char_recognition_processes):
            if process.is_alive():
                process.join()
                print(f"[Process] Character recognition process {idx + 1} stopped.")

        # Clear all queues
        for idx in range(2):
            clear_queue(self.vehicle_result_queues[idx])
            clear_queue(self.plate_result_queues[idx])
            clear_queue(self.img_restoration_result_queues[idx])
            clear_queue(self.text_detection_result_queues[idx])
            clear_queue(self.char_recognize_result_queues[idx])

        print("[Controller] All processes and threads stopped.")

    def post_process_work_thread(self):
        previous_object_id = None

        while True:
            if self.stopped.is_set():
                break

            try:
                for idx in range(len(self.char_recognize_result_queues)):
                    if idx >= len(self.char_recognize_result_queues):
                        break

                    queue = self.char_recognize_result_queues[idx]
                    if not queue.empty():
                        result = queue.get()

                        if result is None:
                            print(f"Queue {idx + 1}: Result is None")
                            continue

                        self._current_result = result
                        object_id = result.get("object_id")
                        floor_id = result.get("floor_id", 0)
                        cam_id = result.get("cam_id", "")
                        car_direction = result.get("car_direction", None)
                        arduino_idx = result.get("arduino_idx", None)
                        start_line = result.get("start_line", None)
                        end_line = result.get("end_line", None)
                        plate_no = result.get("plate_no", None)

                        # print(f'object_id: {object_id}, start_line: {start_line} & end_line: {end_line} & plate_no: {plate_no} & car_direction: {car_direction}')

                        if object_id == previous_object_id:
                            plate_no_data = {
                                "plate_no": plate_no,
                                "floor_id": floor_id,
                                "cam_id": cam_id
                            }
                            self.container_plate_no.append(plate_no_data)
                            print(f'Queue {idx + 1}: plate_no: {plate_no}, object_id: {object_id}')

                        else:
                            if len(self.container_plate_no) > 0:
                                plate_no_list = [data["plate_no"] for data in self.container_plate_no]
                                plate_no_max = most_freq(plate_no_list)
                                plate_no_detected = plate_no_max
                                status_plate_no = check_db(plate_no_detected)

                                plate_no_is_registered = True
                                if not status_plate_no:
                                    logger.write(
                                        f"Warning, plate is unregistered, reading container text!! : {plate_no_detected}",
                                        logger.WARN
                                    )
                                    plate_no_is_registered = False

                                current_max_slot, current_slot_update, current_vehicle_total_update = parking_space_vehicle_counter(
                                    floor_id=floor_id,
                                    cam_id=cam_id,
                                    arduino_idx=arduino_idx,
                                    car_direction=car_direction,
                                    plate_no=plate_no_detected
                                )

                                matrix_update = MatrixController(arduino_idx, max_car=current_max_slot, total_car=current_slot_update)
                                available_space = matrix_update.get_total()
                                self.total_slot = current_max_slot - available_space
                                self.last_result_plate_no = plate_no_detected

                                print(f"PLAT_NO : {plate_no_detected}, AVAILABLE PARKING SPACES : {available_space}, "
                                    f"STATUS : {'TAMBAH' if not car_direction else 'KURANG'}, "
                                    f"VEHICLE_TOTAL: {current_vehicle_total_update}, FLOOR : {floor_id}, "
                                    f"CAMERA : {cam_id}, TOTAL_FRAME: {len(self.container_plate_no)}")

                                self.db_vehicle_history.create_vehicle_history_record(
                                    plate_no=self.last_result_plate_no,
                                    floor_id=floor_id,
                                    camera=cam_id
                                )

                                char = "H" if plate_no_is_registered else "M"
                                matrix_text = f"{plate_no_detected},{char};"
                                # self.matrix_text.write_arduino(matrix_text)

                                # Reset the container after processing
                                self.container_plate_no = []

                                if not self.db_plate.check_exist_plat(plate_no_detected):
                                    plate_no_is_registered = False
                                    logger.write(
                                        f"WARNING THERE IS NO PLATE IN DATABASE!!! text: {plate_no_detected}, status: {car_direction}",
                                        logger.WARNING
                                    )

                            previous_object_id = object_id
                            plate_no_data = {
                                "plate_no": plate_no,
                                "floor_id": floor_id,
                                "cam_id": cam_id
                            }
                            self.container_plate_no.append(plate_no_data)
                            print(f'Queue {idx + 1}: New plate_no: {plate_no}, object_id: {object_id}')


            except Exception as e:
                print(f"Error in post-process work thread: {e}")

    def distribute_vehicle_result(self, vehicle_result):
        if vehicle_result:
            object_id = vehicle_result.get("object_id")
            start_line = vehicle_result.get("start_line", None)
            end_line = vehicle_result.get("end_line", None)

            if start_line and end_line:
                target_queue_index = 1 if object_id % 2 == 0 else 0

                if object_id not in self.object_id_count:
                    self.object_id_count[object_id] = 0

                if self.object_id_count[object_id] < 7:
                    # if object_id != self.previous_object_id:
                    #     print(f"object_id: {object_id}, start_line: {start_line}, end_line: {end_line}, NEW queue: {target_queue_index}")
                    # else:
                    #     print(f"object_id: {object_id}, start_line: {start_line}, end_line: {end_line}, queue: {target_queue_index}")

                    self.vehicle_result_queues[target_queue_index].put(vehicle_result)

                    self.object_id_count[object_id] += 1
                    self.previous_object_id = object_id
                # else:
                #     print(f"object_id: {object_id} has reached the max limit of 7 occurrences.")

    def main(self):
        IS_DEBUG = True
        video_source = config.VIDEO_SOURCE_PC if IS_DEBUG else config.CAM_SOURCE_LT

        # Start processes
        self.start()

        # Initialize and start cameras
        caps = [CameraV1(video, is_video=True) for video in video_source]
        for cap in caps:
            print(f"Starting camera: {cap}")
            cap.start()

        plat_detects = [None] * len(video_source)
        arduino_devices = [Arduino(baudrate=115200, serial_number=serial) for serial in config.SERIALS]
        frames = [None] * len(caps)
        total_slots = {}

        # Set up detection controllers and matrix controllers
        for i, cap in enumerate(caps):
            idx, cam_position = check_floor(i)

            if idx not in total_slots:
                slot = self.db_floor.get_slot_by_id(idx)
                total_slots[idx] = slot["slot"]
                print(f"Slot initialized for idx {idx}: {slot}")

            arduino_text = arduino_devices[idx]
            arduino_matrix = arduino_devices[idx + 1]

            self.matrix_controller = MatrixController(arduino_matrix, max_car=18, total_car=total_slots[idx])
            self.matrix_controller.start(self.matrix_controller.get_total())

            plat_detects[i] = DetectionControllerV5(arduino_text)
            plat_detects[i].start()

        try:
            print("Opening cameras...")
            while True:
                for i, cap in enumerate(caps):
                    idx, cam_position = check_floor(i)
                    ret, frames[i] = cap.read()

                    if not ret:
                        print(f"Failed to read frame from camera {i}")
                        continue

                    if plat_detects[i] is not None:
                        plat_detects[i].process_frame(frames[i], floor_id=idx, cam_id=cam_position)
                        vehicle_result = plat_detects[i].get_vehicle_results()

                        self.distribute_vehicle_result(vehicle_result)

                    # Check if 'q' key is pressed to exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Exit key 'q' pressed. Stopping...")
                        raise KeyboardInterrupt
                # Check if 'q' key is pressed to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit key 'q' pressed. Stopping...")
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            print("Terminating...")

        finally:
            # Clean up and release resources
            self.stop()
            for plat_detect in plat_detects:
                if plat_detect:
                    plat_detect.stop()

            for cap in caps:
                cap.release()

            cv2.destroyAllWindows()
            print("All resources released and program terminated.")


if __name__ == "__main__":
    wrapper = Wrapper()
    wrapper.main()




    # def distribute_vehicle_result(self, vehicle_result):
    #     """Distribute vehicle result based on object_id."""
    #     if vehicle_result:
    #         object_id = vehicle_result.get("object_id")
    #         start_line = vehicle_result.get("start_line", None)
    #         end_line = vehicle_result.get("end_line", None)

    #         if start_line and end_line:
    #             target_queue_index = 1 if object_id % 2 == 0 else 0

    #             if object_id != self.previous_object_id:
    #                 print(f"object_id: {object_id}, NEW queue: {target_queue_index}")
    #                 self.previous_object_id = object_id
    #             else:
    #                 print(f"object_id: {object_id}, queue: {target_queue_index}")

    #             self.vehicle_result_queues[target_queue_index].put(vehicle_result)

        # Additional check to process any remaining data in the queues
        # self.process_remaining_data()

    # def process_remaining_data(self):
    #     """Process remaining data in the result queues after stopping."""
    #     for idx in range(len(self.char_recognize_result_queues)):
    #         queue = self.char_recognize_result_queues[idx]
    #         while not queue.empty():
    #             result = queue.get()
    #             if result is None:
    #                 print(f"Queue {idx + 1}: Result is None")
    #                 continue

    #             self.handle_result(result)

    # def handle_result(self, result):
    #     """Handles the result from character recognition."""
    #     object_id = result.get("object_id")
    #     floor_id = result.get("floor_id", 0)
    #     cam_id = result.get("cam_id", "")
    #     car_direction = result.get("car_direction", None)
    #     arduino_idx = result.get("arduino_idx", None)
    #     start_line = result.get("start_line", None)
    #     end_line = result.get("end_line", None)
    #     plate_no = result.get("plate_no", None)

    #     # Append plate_no if both start_line and end_line are True
    #     if start_line and end_line and plate_no is not None:
    #         plate_no_data = {
    #             "plate_no": plate_no,
    #             "floor_id": floor_id,
    #             "cam_id": cam_id
    #         }
    #         self.container_plate_no.append(plate_no_data)
    #         print(f"Detected plate_no: {plate_no}")

    #     # Process plate numbers when both lines are False
    #     if not start_line and not end_line:
    #         self.process_collected_plate_numbers()

    # def process_collected_plate_numbers(self):
    #     """Processes collected plate numbers when both start and end lines are False."""
    #     if len(self.container_plate_no) > 0:
    #         plate_no_list = [data["plate_no"] for data in self.container_plate_no]
    #         plate_no_max = most_freq(plate_no_list)
    #         plate_no_detected = plate_no_max
    #         status_plate_no = check_db(plate_no_detected)

    #         plate_no_is_registered = True
    #         if not status_plate_no:
    #             logger.write(
    #                 f"Warning, plate is unregistered, reading container text!! : {plate_no_detected}",
    #                 logger.WARN
    #             )
    #             plate_no_is_registered = False

    #         # Update vehicle count and matrix
    #         self.update_vehicle_count_and_matrix(plate_no_detected)

    #         # Reset the container after processing
    #         self.container_plate_no = []

    # def update_vehicle_count_and_matrix(self, plate_no_detected):
    #     """Updates vehicle count and matrix based on detected plate number."""
    #     current_max_slot, current_slot_update, current_vehicle_total_update = parking_space_vehicle_counter(
    #         floor_id=self.floor_id,
    #         cam_id=self.cam_id,
    #         arduino_idx=self.arduino_idx,
    #         car_direction=self.car_direction,
    #         plate_no=plate_no_detected
    #     )

    #     matrix_update = MatrixController(self.arduino_idx, max_car=current_max_slot, total_car=current_slot_update)
    #     available_space = matrix_update.get_total()
    #     self.total_slot = current_max_slot - available_space
    #     self.last_result_plate_no = plate_no_detected

    #     print(f"PLAT_NO : {plate_no_detected}, AVAILABLE PARKING SPACES : {available_space}, "
    #           f"STATUS : {'TAMBAH' if not self.car_direction else 'KURANG'}, "
    #           f"VEHICLE_TOTAL: {current_vehicle_total_update}, FLOOR : {self.floor_id}, "
    #           f"CAMERA : {self.cam_id}, TOTAL_FRAME: {len(self.container_plate_no)}")

    #     self.db_vehicle_history.create_vehicle_history_record(
    #         plate_no=self.last_result_plate_no,
    #         floor_id=self.floor_id,
    #         camera=self.cam_id
    #     )

    #     char = "H" if self.plate_no_is_registered else "M"
    #     matrix_text = f"{plate_no_detected},{char};"
    #     # self.matrix_text.write_arduino(matrix_text)