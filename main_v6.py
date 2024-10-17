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
        self.plate_detection_processes = None
        self.image_restoration_processes = None
        self.text_detection_processes = None
        self.char_recognition_processes = None
        self.vehicle_bounding_boxes = []
        self.floor_id = 0
        self.cam_id = ""
        self._current_frame = None
        self._current_result = None
        self.result_processing_thread = None
        self.container_plate_no = []
        self.db_plate = PlatController()
        self.db_floor = FloorController()
        self.db_mysn = FetchAPIController()
        self.db_vehicle_history = VehicleHistoryController()

    def start(self):
        print("[Thread] Starting result processing thread...")
        self.result_processing_thread = threading.Thread(target=self.post_process_work_thread)
        self.result_processing_thread.start()

        for idx in range(2):
            self.start_detection_processes(idx)

    def start_detection_processes(self, idx):
        print(f"[Process] Starting plate detection process for Queue {idx + 1}...")
        plate_detection_processes = mp.Process(
            target=plate_detection,
            args=(self.stopped, self.vehicle_result_queues[idx], self.plate_result_queues[idx])
        )
        # self.processes.append(plate_detection_process)
        plate_detection_processes.start()

        print(f"[Process] Starting image restoration process for Queue {idx + 1}...")
        image_restoration_processes = mp.Process(
            target=image_restoration,
            args=(self.stopped, self.plate_result_queues[idx], self.img_restoration_result_queues[idx])
        )
        # self.processes.append(image_restoration)
        image_restoration_processes.start()

        print(f"[Process] Starting text detection process for Queue {idx + 1}...")
        text_detection_processes = mp.Process(
            target=text_detection,
            args=(self.stopped, self.img_restoration_result_queues[idx], self.text_detection_result_queues[idx])
        )
        # self.processes.append(text_detection)
        text_detection_processes.start()

        print(f"[Process] Starting character recognition process for Queue {idx + 1}...")
        char_recognition_processes = mp.Process(
            target=character_recognition,
            args=(self.stopped, self.text_detection_result_queues[idx], self.char_recognize_result_queues[idx])
        )
        # self.processes.append(char_recognition)
        char_recognition_processes.start()

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
        while True:
            if self.stopped.is_set():
                break

            try:
                # Iterate through each queue in char_recognize_result_queues
                for queue in self.char_recognize_result_queues:
                    if not queue.empty():
                        result = queue.get()

                        if result is None:
                            print("Result is None", result)
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

                        # Print object details for debugging
                        # print(f'object_id: {object_id}, start_line: {start_line} & end_line: {end_line} & plate_no: {plate_no} & car_direction: {car_direction}')

                        # Append plate_no if start_line and end_line are both True
                        if start_line and end_line and plate_no is not None:
                            plate_no_data = {
                                "plate_no": plate_no,
                                "floor_id": floor_id,
                                "cam_id": cam_id
                            }
                            self.container_plate_no.append(plate_no_data)
                            print(f'plate_no: {plate_no}')

                        # If both start_line and end_line are False, process the collected plate numbers
                        if not start_line and not end_line:
                            if len(self.container_plate_no) > 0:
                                plate_no_list = [data["plate_no"] for data in self.container_plate_no]

                                plate_no_max = most_freq(plate_no_list)
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

                                char = "H" if plate_no_is_registered else "M"
                                matrix_text = f"{plate_no_detected},{char};"
                                # self.matrix_text.write_arduino(matrix_text)
                                self.container_plate_no = []

                                if not self.db_plate.check_exist_plat(plate_no_detected):
                                    plate_no_is_registered = False
                                    logger.write(
                                        f"WARNING THERE IS NO PLAT IN DATABASE!!! text: {plate_no_detected}, status: {car_direction}",
                                        logger.WARNING
                                    )
            except Exception as e:
                print(f"Error in post-process work thread: {e}")


    def process_result(self, result):
        object_id = result.get("object_id")
        floor_id = result.get("floor_id", 0)
        cam_id = result.get("cam_id", "")
        car_direction = result.get("car_direction", None)
        arduino_idx = result.get("arduino_idx", None)
        start_line = result.get("start_line", None)
        end_line = result.get("end_line", None)
        plate_no = result.get("plate_no", None)

        # Append plate_no if start_line and end_line are both True
        if start_line and end_line and plate_no is not None:
            plate_no_data = {
                "plate_no": plate_no,
                "floor_id": floor_id,
                "cam_id": cam_id
            }
            self.container_plate_no.append(plate_no_data)
            print(f'plate_no: {plate_no}')

        # If both start_line and end_line are False, process the collected plate numbers
        if not start_line and not end_line:
            self.process_collected_plate_numbers(floor_id, cam_id, arduino_idx, car_direction)

    def process_collected_plate_numbers(self, floor_id, cam_id, arduino_idx, car_direction):
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

            print(f"PLATE_NO : {plate_no_detected}, AVAILABLE PARKING SPACES : {available_space}, STATUS : {'ADD' if not car_direction else 'REMOVE'}, VEHICLE_TOTAL: {current_vehicle_total_update}, FLOOR : {floor_id}, CAMERA : {cam_id}, TOTAL_FRAME: {len(self.container_plate_no)}")

            self.db_vehicle_history.create_vehicle_history_record(plate_no=self.last_result_plate_no, floor_id=floor_id, camera=cam_id)

            char = "H" if plate_no_is_registered else "M"
            matrix_text = f"{plate_no_detected},{char};"
            # self.matrix_text.write_arduino(matrix_text)
            self.container_plate_no = []

            if not self.db_plate.check_exist_plat(plate_no_detected):
                plate_no_is_registered = False
                logger.write(
                    f"WARNING: THERE IS NO PLATE IN DATABASE!!! text: {plate_no_detected}, status: {car_direction}",
                    logger.WARNING
                )

    def char_data_parsed(self, last_result):
        object_id = last_result.get("object_id")
        plate_no = last_result.get("plate_no", "")
        bg_color = last_result.get("bg_color", None)
        floor_id = last_result.get('floor_id', 0)
        cam_id = last_result.get('cam_id', "")
        arduino_idx = last_result.get('arduino_idx')
        car_direction = last_result.get('car_direction')
        start_line = last_result.get('start_line', False)
        end_line = last_result.get('end_line', False)

        if plate_no:
            result = {
                "object_id": object_id,
                "plate_no": plate_no,
                "bg_color": bg_color,
                "frame": "",
                "floor_id": floor_id,
                "cam_id": cam_id,
                "arduino_idx": str(arduino_idx),
                "car_direction": car_direction,
                "start_line": start_line,
                "end_line": end_line
            }
            print(json.dumps(result, indent=4))
        # else:
        #     print("No valid plate number detected.")

        # print(f"Processed Plate Detection Result: {char_recognize_result_queue}")

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

                        if vehicle_result:
                            object_id = vehicle_result.get("object_id")
                            
                            # Route vehicle result to the appropriate queue
                            target_queue = self.vehicle_result_queues[object_id % 2]
                            target_queue.put(vehicle_result)
                            # print(f"Object ID {object_id} sent to {'Queue 1' if object_id % 2 == 0 else 'Queue 2'}")

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