import os, sys
import cv2
import threading
import multiprocessing as mp
import numpy as np
import time
import re

# src_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
# sys.path.append(src_path)
# print("this_path: ", src_path)

from src.Integration.arduino import Arduino
from src.Integration.newArduino import *
from src.config.config import config
from src.controllers.detection_controller import DetectionController
from src.models.cam_model import CameraV1

from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.vehicle_history_controller import VehicleHistoryController
from src.utils.multiprocessing_util import put_queue_none, clear_queue
from src.models.image_restoration_model import image_restoration, ImageRestoration
from src.models.text_detection_model import text_detection
from src.models.character_recognition_model import character_recognition, ModelAndLabelLoader

from src.config.logger import Logger
from src.utils.util import (
    most_freq, 
    check_db, 
    parking_space_vehicle_counter,
    most_freq, 
    draw_tracking_points,
    define_tracking_polygon,
    send_plate_data,
    response_post
)
from src.utils.plate_util import (
    match_plate_no,
    process_plate_data
)
from src.view.display import (
    add_overlay,
    create_grid
)

logger = Logger("main", is_save=True)


class Wrapper:
    def __init__(self) -> None:
        self.IS_DEBUG = config.IS_DEBUG
        self.IS_VIDEO = config.IS_VIDEO
        self.IS_PC = config.IS_PC
        self.previous_object_id = None
        self.num_processes = 1
        self.object_id_count = {}
        self.vehicle_plate_result_queue_list = [mp.Queue() for _ in range(self.num_processes)]
        self.plate_result_queues = [mp.Queue() for _ in range(self.num_processes)]
        self.img_restoration_result_queues = [mp.Queue() for _ in range(self.num_processes)]
        self.text_detection_result_queues = [mp.Queue() for _ in range(self.num_processes)]
        self.char_recognize_result_queues = [mp.Queue() for _ in range(self.num_processes)]
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
        self.max_num_frame = 1
        self._current_frame = None
        self._current_result = None
        self.result_processing_thread = None
        self.db_floor = FloorController()
        self.db_vehicle_history = VehicleHistoryController()

        self.queue_index = 0
        self.clicked_points = []

        self._model_built_events = []
        self.vehicle_plate_result_queue = mp.Queue()

        self.BASE_DIR = config.BASE_PC_DIR if self.IS_PC else config.BASE_LAPTOP_DIR
        self.MODEL_VEHICLE_PLATE_PATH = os.path.join(self.BASE_DIR, config.MODEL_VEHICLE_PLATE_PATH)

        if not self.IS_DEBUG:
            res_post = send_plate_data(floor_id="1", plate_no="BP1234BP", cam_position="in")

            self.arduino_devices = [Arduino(baudrate=115200, com=None) for com in config.SERIAL_COM]
            response_post(res_post, self.arduino_devices)
            # for ard, com in zip(self.arduino_devices, config.SERIAL_COM):
            #    ard.write(11, com)
        else:
            self.arduino_devices = [Arduino(baudrate=115200, com=None) for com in config.SERIAL_COM]

    def start(self):
        print("[Thread] Starting result processing thread...")
        self.result_processing_thread = threading.Thread(target=self.post_process_work_thread)
        self.result_processing_thread.start()

        self.distribute_thread_handle = threading.Thread(target=self.distribute_work_thread)
        self.distribute_thread_handle.start()

        for idx in range(self.num_processes):
            self.start_detection_processes(idx)

    def start_detection_processes(self, idx):
        print(f"[Process] Starting image restoration process for Queue {idx + 1}...")
        model_built_event1 = mp.Event()
        image_restoration_process = mp.Process(
            target=image_restoration,
            args=(self.stopped, self.BASE_DIR, model_built_event1, self.vehicle_plate_result_queue_list[idx], self.img_restoration_result_queues[idx])
        )
        image_restoration_process.start()
        self.image_restoration_processes.append(image_restoration_process)
        self._model_built_events.append(model_built_event1)

        print(f"[Process] Starting text detection process for Queue {idx + 1}...")
        model_built_event2 = mp.Event()
        text_detection_process = mp.Process(
            target=text_detection,
            args=(self.stopped, model_built_event2, self.img_restoration_result_queues[idx], self.text_detection_result_queues[idx])
        )
        text_detection_process.start()
        self.text_detection_processes.append(text_detection_process)
        self._model_built_events.append(model_built_event2)

        print(f"[Process] Starting character recognition process for Queue {idx + 1}...")
        model_built_event3 = mp.Event()
        char_recognition_process = mp.Process(
            target=character_recognition,
            args=(self.stopped, self.BASE_DIR, model_built_event3, self.text_detection_result_queues[idx], self.char_recognize_result_queues[idx])
        )
        char_recognition_process.start()
        self.char_recognition_processes.append(char_recognition_process)
        self._model_built_events.append(model_built_event3)

    def is_model_built(self):
        return all([event.is_set() for event in self._model_built_events])

    def stop(self):
        print("[Controller] Stopping detection processes and threads...")
        self.stopped.set()

        # Put None in all queues to signal termination
        for idx in range(self.num_processes):
            put_queue_none(self.vehicle_plate_result_queue_list[idx])
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
        for idx in range(self.num_processes):
            clear_queue(self.vehicle_plate_result_queue_list[idx])
            clear_queue(self.plate_result_queues[idx])
            clear_queue(self.img_restoration_result_queues[idx])
            clear_queue(self.text_detection_result_queues[idx])
            clear_queue(self.char_recognize_result_queues[idx])

        print("[Controller] All processes and threads stopped.")

    def post_process_work_thread(self):
        previous_object_id = None
        prev_floor_id = None
        prev_cam_id = None
        object_id_count = 0

        while True:
            if self.stopped.is_set():
                break

            try:
                for idx, queue in enumerate(self.char_recognize_result_queues):
                    if queue.empty():
                        continue

                    result = queue.get()
                    if result is None:
                        print(f"Queue {idx + 1}: Result is None")
                        continue

                    self._current_result = result
                    object_id = result.get("object_id")
                    floor_id = result.get("floor_id")
                    cam_id = result.get("cam_id")
                    car_direction = result.get("car_direction")
                    arduino_idx = result.get("arduino_idx")
                    plate_no = result.get("plate_no")
                    plate_no_easyocr = result.get("plate_no_easyocr") 

                    if self.max_num_frame == 1:
                        self.container_plate_no.append({
                            "plate_no": plate_no,
                            "plate_no_easyocr": plate_no_easyocr,
                            "floor_id": floor_id,
                            "cam_id": cam_id
                        })

                        send_plate_no, response_api_counter = process_plate_data(
                            floor_id=floor_id, 
                            cam_id=cam_id, 
                            arduino_idx=arduino_idx, 
                            car_direction=car_direction, 
                            container_plate_no=self.container_plate_no
                        )

                        if not response_api_counter:
                            print("response_api_counter: ", response_api_counter)

                        unoccupied = response_post(response_api_counter, self.arduino_devices)

                        self.container_plate_no = []

                        # if not self.IS_DEBUG:
                        #     response_post(response_api_counter, self.arduino_devices)
                        # else:
                        #     print("SEND DATA IS SUCCESS - IS_DEBUG")

                    else:
                        if (object_id == previous_object_id and
                            floor_id == prev_floor_id and
                            cam_id == prev_cam_id):
                            
                            object_id_count += 1

                            self.container_plate_no.append({
                                "plate_no": plate_no,
                                "plate_no_easyocr": plate_no_easyocr,
                                "floor_id": floor_id,
                                "cam_id": cam_id
                            })

                            if object_id_count == self.max_num_frame:
                                send_plate_no, response_api_counter = process_plate_data(
                                    floor_id=floor_id, 
                                    cam_id=cam_id, 
                                    arduino_idx=arduino_idx, 
                                    car_direction=car_direction, 
                                    container_plate_no=self.container_plate_no
                                )

                                if not response_api_counter:
                                    print("response_api_counter: ", response_api_counter)

                                unoccupied = response_post(response_api_counter, self.arduino_devices)

                                self.container_plate_no = []
                                object_id_count = 0

                        else:
                            if self.container_plate_no:
                                print(f"Storing data for different object. Previous: {previous_object_id}, New: {object_id}")

                            object_id_count = 1                        
                            previous_object_id = object_id
                            prev_floor_id = floor_id
                            prev_cam_id = cam_id

                            self.container_plate_no.append({
                                "plate_no": plate_no,
                                "plate_no_easyocr": plate_no_easyocr,
                                "floor_id": floor_id,
                                "cam_id": cam_id
                            })

                    logger.write(
                        f"Queue {idx + 1}, ID: {object_id}, PLATE_NO: {plate_no}, PLATE_NO_EASYOCR: {plate_no_easyocr}, SEND_PLATE_NO: {send_plate_no}, {'TAMBAH' if not car_direction else 'KURANG'}, FLOOR : {floor_id}, CAMERA : {cam_id}, UNOCCUPIED: {unoccupied}",
                        logger.DEBUG
                    )

            except Exception as e:
                print(f"Error in post-process work thread: {e}")

    def distribute_work_thread(self):
        while True:
            if self.stopped.is_set():
                break

            try:
                vehicle_plate_data = self.vehicle_plate_result_queue.get()
 
                sizes = [q.qsize() for q in self.vehicle_plate_result_queue_list]
                sorted_indices = np.argsort(sizes)
                smallest_size_id = sorted_indices[0]
                self.vehicle_plate_result_queue_list[smallest_size_id].put(vehicle_plate_data)
            except Exception as e:
                print("Error at distribute_work_thread", e)

    def main(self):
        if self.IS_DEBUG:
            ROWS_CONF, COLS_CONF = 1, 1
            FLOOR_CAM_CONF = {
                2: {"IN": False, "OUT": False},
                3: {"IN": True, "OUT": False},
                4: {"IN": False, "OUT": False},
                5: {"IN": False, "OUT": False},
            }

        else:
            ROWS_CONF, COLS_CONF = 2, 3
            FLOOR_CAM_CONF = {
                2: {"IN": True, "OUT": True},
                3: {"IN": True, "OUT": True},
                4: {"IN": True, "OUT": True},
                5: {"IN": False, "OUT": False},
            }

        frame_size_options = {
            (1, 1): 1280,
            (1, 2): 720,
            (1, 3): 640,
            (2, 1): 640,
            (2, 2): 640,
            (2, 3): 640
        }
        FRAME_SIZE_CONF = frame_size_options.get((ROWS_CONF, COLS_CONF), 640)

        VIDEO_PATH = config.VIDEO_SOURCE_FLOOR_PC if self.IS_DEBUG and self.IS_PC else (
            config.VIDEO_SOURCE_FLOOR_LAPTOP if self.IS_DEBUG else config.CAM_SOURCE_FLOOR
        )

        active_sources = [
            (floor_id, cam_id, source)
            for floor_id, cams in VIDEO_PATH.items()
            for cam_id, source in cams.items()
            if FLOOR_CAM_CONF.get(floor_id, {}).get(cam_id, False)
        ]

        self.start()

        while not self.is_model_built():
            time.sleep(0.1)
            print("Loading recognition models...")

        plat_detects = [None] * len(active_sources)
        frames = [None] * len(active_sources)

        for i, (idx, cam_id, source) in enumerate(active_sources):          
            plat_detects[i] = DetectionController(arduino_matrix="", vehicle_plate_result_queue=self.vehicle_plate_result_queue, base_dir=self.BASE_DIR)
            plat_detects[i].start()
        
        while not all([m.is_model_built() for m in plat_detects]):
            time.sleep(0.1)
            print("Loading detection models...")

        caps = [CameraV1(source, is_video=self.IS_VIDEO) for _, _, source in active_sources]
        for cap in caps:
            print(f"Starting camera: {cap}")
            cap.start()

        try:
            print("Opening cameras...")
            while True:
                try:
                    for i, (floor_id, cam_id, source) in enumerate(active_sources):
                        # print("floor_id, cam_id: ", floor_id, cam_id)
                        num, frames[i] = caps[i].read()

                        if frames[i] is None:
                            print(f"frame {i} is None", frames[i])
                            frames[i] = np.zeros((1440, 2560, 3), dtype=np.uint8)
                            caps[i].open()
                            continue

                        plat_detects[i].process_frame({
                            "frame": frames[i].copy(),
                            "floor_id": floor_id,
                            "cam_id": cam_id,
                        })

                        # frames[i] = cv2.resize(frames[i], (1080, 720))
                        height, width = frames[i].shape[:2]
    
                        slot = self.db_floor.get_slot_by_id(floor_id)
                        total_slot, vehicle_total = slot["slot"], slot["vehicle_total"]
    
                        poly_points, tracking_points, poly_bbox = define_tracking_polygon(
                            height=height, 
                            width=width, 
                            floor_id=floor_id, 
                            cam_id=cam_id
                        )

                        draw_tracking_points(frames[i], tracking_points, (height, width))

                        last_plate_no = self.db_vehicle_history.get_vehicle_history_by_floor_id(floor_id)["plate_no"]
                        plate_no = last_plate_no if last_plate_no else ""

                        add_overlay(frames[i], floor_id, cam_id, poly_points, plate_no, total_slot, vehicle_total, poly_bbox=poly_bbox)

                    frame_show = create_grid(frames, rows=ROWS_CONF, cols=COLS_CONF, frame_size=FRAME_SIZE_CONF, padding=5)
                    cv2.imshow("Camera", frame_show)

                except Exception as e:
                    print("Error: get_frame from process_frame", e)

                # Check jika tombol 'q' ditekan untuk keluar
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit key 'q' pressed. Stopping...")
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            print("Terminating...")

        finally:
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


    # def post_process_work_thread(self):
    #     previous_object_id = None
    #     prev_floor_id = None
    #     prev_cam_id = None
    #     object_id_count = 0

    #     while True:
    #         if self.stopped.is_set():
    #             break

    #         try:
    #             for idx in range(len(self.char_recognize_result_queues)):
    #                 if idx >= len(self.char_recognize_result_queues):
    #                     break

    #                 queue = self.char_recognize_result_queues[idx]
    #                 if not queue.empty():
    #                     result = queue.get()

    #                     if result is None:
    #                         print(f"Queue {idx + 1}: Result is None")
    #                         continue

    #                     # Extract data from the result
    #                     self._current_result = result
    #                     object_id = result.get("object_id")
    #                     floor_id = result.get("floor_id")
    #                     cam_id = result.get("cam_id")
    #                     car_direction = result.get("car_direction")
    #                     arduino_idx = result.get("arduino_idx")
    #                     plate_no = result.get("plate_no")

    #                     if (
    #                         object_id == previous_object_id and
    #                         floor_id == prev_floor_id and
    #                         cam_id == prev_cam_id
    #                     ):
    #                         object_id_count += 1  # Increment count
    #                         print(f"object_id_count ====== : {object_id_count}")

    #                         # Add plate number data to the container
    #                         self.container_plate_no.append({
    #                             "plate_no": plate_no,
    #                             "floor_id": floor_id,
    #                             "cam_id": cam_id
    #                         })

    #                         print(f'Queue {idx + 1}: plate_no: {plate_no}, object_id: {object_id}')

    #                         if object_id_count == self.max_num_frame:
    #                             self.process_plate_data(floor_id, cam_id, arduino_idx, car_direction)

    #                     else:
    #                         # Process the collected plate data if available
    #                         if self.container_plate_no:
    #                             self.process_plate_data(floor_id, cam_id, arduino_idx, car_direction)

    #                         # Reset and store the new object details
    #                         previous_object_id = object_id
    #                         prev_floor_id = floor_id
    #                         prev_cam_id = cam_id
    #                         object_id_count = 1  # Reset to 1 (new object)

    #                         # Add initial plate number data
    #                         self.container_plate_no = [{
    #                             "plate_no": plate_no,
    #                             "floor_id": floor_id,
    #                             "cam_id": cam_id
    #                         }]

    #                         print(f'Queue {idx + 1}: New plate_no: {plate_no}, object_id: {object_id}')

    #         except Exception as e:
    #             print(f"Error in post-process work thread: {e}")

    # def process_plate_data(self, floor_id, cam_id, arduino_idx, car_direction):
    #     """
    #     Processes plate number data and updates the parking status.
    #     """
    #     plate_no_list = [data["plate_no"] for data in self.container_plate_no]
    #     plate_no_max = most_freq(plate_no_list)
    #     status_plate_no = check_db(plate_no_max)

    #     plate_no_is_registered = True
    #     if not status_plate_no:
    #         logger.write(
    #             f"Warning, plate is unregistered, reading container text!! : {plate_no_max}",
    #             logger.WARN
    #         )
    #         plate_no_is_registered = False

    #     parking_space_vehicle_counter(
    #         floor_id=floor_id,
    #         cam_id=cam_id,
    #         arduino_idx=arduino_idx,
    #         car_direction=car_direction,
    #         plate_no=plate_no_max,
    #         container_plate_no=self.container_plate_no,
    #         plate_no_is_registered=plate_no_is_registered
    #     )

    #     self.container_plate_no = []
    #     print("Processed plate data and cleared the container.")