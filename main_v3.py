import os
import cv2
import threading
import multiprocessing as mp
import json
from ultralytics import YOLO

from src.Integration.arduino import Arduino
from src.config.config import config
from src.controllers.matrix_controller import MatrixController
from src.controllers.detection_controller_v3 import DetectionControllerV3
from src.models.cam_model import CameraV1
from src.controllers.utils.util import check_floor
from src.view.show_cam import show_cam, show_text, show_line
from src.Integration.service_v1.controller.plat_controller import PlatController
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.fetch_api_controller import FetchAPIController
from src.Integration.service_v1.controller.vehicle_history_controller import VehicleHistoryController
from utils.multiprocessing_util import put_queue_none, clear_queue
from src.models.plate_detection_model_v3 import plate_detection_process
from src.models.image_restoration_model_v3 import image_restoration
from src.models.text_detection_model_v3 import text_detection
from src.models.character_recognition_model_v3 import character_recognition, ModelAndLabelLoader
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
        self.container_plate_no = []
        self.db_plate = PlatController()
        self.db_floor = FloorController()
        self.db_mysn = FetchAPIController()
        self.db_vehicle_history = VehicleHistoryController()
    
    # def load_model(self):
    #     plate_model_path = config.MODEL_PATH_PLAT_v2
    #     plate_model = YOLO(plate_model_path)

    #     char_model_path = config.MODEL_CHAR_RECOGNITION_PATH
    #     char_weight_path = config.WEIGHT_CHAR_RECOGNITION_PATH
    #     label_path = config.LABEL_CHAR_RECOGNITION_PATH

    #     char_model = ModelAndLabelLoader.load_model(char_model_path, char_weight_path)
    #     char_label = ModelAndLabelLoader.load_labels(label_path)

    def start(self):
        print("[Thread] Starting result processing thread...")
        self.result_processing_thread = threading.Thread(target=self.post_process_work_thread)
        self.result_processing_thread.start()

        print("[Process] Starting plate detection process...")
        self.plate_detection_process = mp.Process(
            target=plate_detection_process,
            args=(self.stopped, self.vehicle_result_queue, self.plate_result_queue)
        )
        self.plate_detection_process.start()

        print("[Process] Starting image restoration process...")
        self.image_restoration_process = mp.Process(
            target=image_restoration,
            args=(self.stopped, self.plate_result_queue, self.img_restoration_result_queue)
        )
        self.image_restoration_process.start()

        print("[Process] Starting text detection process...")
        self.text_detection_process = mp.Process(
            target=text_detection,
            args=(self.stopped, self.img_restoration_result_queue, self.text_detection_result_queue)
        )
        self.text_detection_process.start()

        print("[Process] Starting character recognition process...")
        self.char_recognition_process = mp.Process(
            target=character_recognition,
            args=(self.stopped, self.text_detection_result_queue, self.char_recognize_result_queue)
        )
        self.char_recognition_process.start()

    def stop(self):
        print("[Controller] Stopping detection processes and threads...")
        self.stopped.set()
        
        put_queue_none(self.vehicle_result_queue)
        put_queue_none(self.plate_result_queue)
        put_queue_none(self.img_restoration_result_queue)
        put_queue_none(self.text_detection_result_queue)
        put_queue_none(self.char_recognize_result_queue)

        # Stop processes
        if self.result_processing_thread is not None:
            self.result_processing_thread.join()

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
                car_direction = result.get("car_direction", None)
                arduino_idx = result.get("arduino_idx", None)
                start_line = result.get("start_line", None)
                end_line = result.get("end_line", None)
                plate_no = result.get("plate_no", None)

                # print(f'start_line: {start_line} & end_line: {end_line} & plate_no: {plate_no} & car_direction: {car_direction}')

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
                        # print(f'self.container_plate_no: {self.container_plate_no}')
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
                        
                        # self.send_plate_data(floor_id=current_floor_position, plate_no=plate_no, cam_position=current_cam_position)

                        # print('=' * 30 + " BORDER: LAST RESULT " + '=' * 30)

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

    def char_data_parsed(self, last_result):
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

        if IS_DEBUG:
            video_source = config.VIDEO_SOURCE_PC
        else:
            video_source = config.CAM_SOURCE_LT

        # Start the processes
        self.start()

        # Initialize camera captures
        caps = [CameraV1(video, is_video=True) for video in video_source]

        # Start the cameras
        for cap in caps:
            print("Starting camera: ", cap)
            cap.start()

        plat_detects = [None for _ in range(len(video_source))]
        arduino = [Arduino(baudrate=115200, serial_number=serial) for serial in config.SERIALS]
        frames = [None for _ in range(len(caps))]
        total_slots = {}

        for i in range(len(caps)):
            idx, cam_position = check_floor(i)

            if idx not in total_slots:
                slot = self.db_floor.get_slot_by_id(idx)
                print(f'idx {idx}, slot {slot}')
                total_slot = slot["slot"]
                total_slots[idx] = total_slot

            arduino_index = idx
            arduino_text = arduino[arduino_index]
            arduino_matrix = arduino[arduino_index + 1]

            self.matrix_controller = MatrixController(arduino_matrix=arduino_matrix, max_car=18, total_car=total_slot)
            self.matrix_controller.start(self.matrix_controller.get_total())
            plat_detects[i] = DetectionControllerV3(arduino_text)
            plat_detects[i].start()

        try:
            print("Going To Open Cam")
            while True:
                for i in range(len(caps)):
                    idx, cam_position = check_floor(i)
                    ret, frames[i] = caps[i].read()

                    if not ret:
                        print(f"Failed to read frame from camera {i}")
                        continue

                    if plat_detects[i] is not None:
                        plat_detects[i].process_frame(frames[i], floor_id=idx, cam_id=cam_position)
                        results = plat_detects[i].get_vehicle_results()

                        if results:
                            self.vehicle_result_queue.put(results)

                    # Process the output from the plate detection queue
                    if not self.char_recognize_result_queue.empty():
                        last_result = self.char_recognize_result_queue.get()
                        # self.char_data_parsed(last_result=last_result)

                    # Check if 'q' key is pressed to exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Exit key 'q' pressed. Stopping...")
                        raise KeyboardInterrupt

        except KeyboardInterrupt:
            print("Terminating...")

        finally:
            # Stop everything
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