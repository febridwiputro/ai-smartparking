import os
import cv2
import threading
import multiprocessing as mp
import json
from ultralytics import YOLO
import numpy as np
import time

from src.Integration.arduino import Arduino
from src.config.config import config
from src.controllers.matrix_controller import MatrixController
from src.controllers.detection_controller_v6 import DetectionControllerV6
from src.models.cam_model import CameraV1
from src.controllers.utils.util import check_floor
from src.view.show_cam import show_cam, show_text, show_line
from src.Integration.service_v1.controller.plat_controller import PlatController
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.fetch_api_controller import FetchAPIController
from src.Integration.service_v1.controller.vehicle_history_controller import VehicleHistoryController
from utils.multiprocessing_util import put_queue_none, clear_queue
from src.models.image_restoration_model_v6 import image_restoration
from src.models.text_detection_model_v6 import text_detection
from src.models.character_recognition_model_v6 import character_recognition, ModelAndLabelLoader
from src.config.logger import logger
from src.view.show_cam import resize_image
from src.controllers.utils.util import (
    convert_bbox_to_decimal, 
    convert_decimal_to_bbox, 
    crop_frame, 
    most_freq, 
    find_closest_strings_dict, 
    check_db, 
    parking_space_vehicle_counter,
    print_normalized_points,
    most_freq, 
    add_overlay,
    draw_points_and_lines,
    draw_tracking_points
)
from src.controllers.utils.display import draw_box


def create_grid(frames, rows, cols, frame_size=None, padding=5, bg_color=(0, 0, 0)):
    """
    Create a grid of frames (images).

    Parameters:
    - frames (list): List of frames (images) in BGR format.
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.
    - frame_size (tuple): Optional (width, height) to resize each frame.
    - padding (int): Space between frames in pixels.
    - bg_color (tuple): Background color for padding (BGR).

    Returns:
    - grid (np.ndarray): Final grid image.
    """
    expected_num_frames = rows * cols
    current_num_frames = len(frames)
    if current_num_frames < expected_num_frames:
        for i in range(expected_num_frames - current_num_frames):
            frames.append(np.zeros_like(frames[0]))
    elif current_num_frames > expected_num_frames:
        raise ValueError(f"Expected {rows * cols} frames, but got {len(frames)}.")

    # Resize frames if needed
    if frame_size:
        frames = [resize_image(frame, frame_size, frame_size) for frame in frames]

    # Get frame dimensions
    height, width, _ = frames[0].shape

    # Create an empty grid with padding
    grid_height = rows * height + (rows - 1) * padding
    grid_width = cols * width + (cols - 1) * padding
    grid = np.full((grid_height, grid_width, 3), bg_color, dtype=np.uint8)

    # Populate the grid with frames
    for i in range(rows):
        for j in range(cols):
            y = i * (height + padding)
            x = j * (width + padding)
            frame = frames[i * cols + j]
            grid[y:y + height, x:x + width] = frame

    return grid


class Wrapper:
    def __init__(self) -> None:
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
        self._current_frame = None
        self._current_result = None
        self.result_processing_thread = None
        self.db_plate = PlatController()
        self.db_floor = FloorController()
        self.db_mysn = FetchAPIController()
        self.db_vehicle_history = VehicleHistoryController()
        
        self.queue_index = 0
        self.clicked_points = []

        self._model_built_events = []
        self.vehicle_plate_result_queue = mp.Queue()

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
            args=(self.stopped, model_built_event1, self.vehicle_plate_result_queue_list[idx], self.img_restoration_result_queues[idx])
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
            args=(self.stopped, model_built_event3, self.text_detection_result_queues[idx], self.char_recognize_result_queues[idx])
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

                        # Check if the current object matches the previous one, including floor_id and cam_id.
                        if (
                            object_id == previous_object_id and 
                            floor_id == prev_floor_id and 
                            cam_id == prev_cam_id
                        ):
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
                                    plate_no=plate_no_detected,
                                    container_plate_no=self.container_plate_no,
                                    plate_no_is_registered=plate_no_is_registered
                                )

                                # Reset the container after processing
                                self.container_plate_no = []

                            # Update the previous state variables
                            previous_object_id = object_id
                            prev_floor_id = floor_id
                            prev_cam_id = cam_id

                            plate_no_data = {
                                "plate_no": plate_no,
                                "floor_id": floor_id,
                                "cam_id": cam_id
                            }
                            self.container_plate_no.append(plate_no_data)
                            print(f'Queue {idx + 1}: New plate_no: {plate_no}, object_id: {object_id}')

            except Exception as e:
                print(f"Error in post-process work thread: {e}")

    def distribute_work_thread(self):

        while True:
            if self.stopped.is_set():
                break
                
            try:
            
                vehicle_plate_data = self.vehicle_plate_result_queue.get()
    
            # plate_results = []
            # for i in range(len(self.vehicle_plate_result_queue_list_2)):
            #     try:
            #         res = self.vehicle_plate_result_queue_list_2[i].get_nowait()
            #         if res is not None:
            #             plate_results.append(res)
            #     except Exception as e:
            #         pass
            #
            # if len(plate_results) > 0:
                # print("len(plate_results)", len(plate_results))
                    # print(e)
            # print("plate_result", plate_result)

            # if plate_result is None:
            #     continue
            # # print(plate_result)
                sizes = [q.qsize() for q in self.vehicle_plate_result_queue_list]
                sorted_indices = np.argsort(sizes)
                print("sizes", sizes, "sorted_indices", sorted_indices)
                smallest_size_id = sorted_indices[0]
                self.vehicle_plate_result_queue_list[smallest_size_id].put(vehicle_plate_data)
            except Exception as e:
                print("Error at distribute_work_thread", e)

        # print("vehicle_result", vehicle_result)
        # if vehicle_result is not None:
        #     object_id = vehicle_result.get("object_id")
        #     start_line = vehicle_result.get("start_line", None)
        #     end_line = vehicle_result.get("end_line", None)

        #     if start_line and end_line:
        #         # target_queue_index = 1 if object_id % 2 == 0 else 0
        #         target_queue_index = 0

        #         if object_id not in self.object_id_count:
        #             self.object_id_count[object_id] = 0

        #         if self.object_id_count[object_id] < 1:
        #             # if object_id != self.previous_object_id:
        #             #     print(f"object_id: {object_id}, start_line: {start_line}, end_line: {end_line}, NEW queue: {target_queue_index}")
        #             # else:
        #             #     print(f"object_id: {object_id}, start_line: {start_line}, end_line: {end_line}, queue: {target_queue_index}")

        #             print("vehicle_result", vehicle_result)
        #             print(self.vehicle_plate_result_queue_list[target_queue_index].qsize())

        #             self.vehicle_plate_result_queue_list[target_queue_index].put(vehicle_result)

        #             self.object_id_count[object_id] += 1
        #             self.previous_object_id = object_id
                # else:
                #     print(f"object_id: {object_id} has reached the max limit of 7 occurrences.")


    def distribute_vehicle_result(self, vehicle_result):
        # print("vehicle_result", vehicle_result)
        if vehicle_result is not None:
            object_id = vehicle_result.get("object_id")
            start_line = vehicle_result.get("start_line", None)
            end_line = vehicle_result.get("end_line", None)

            if start_line and end_line:
                # target_queue_index = 1 if object_id % 2 == 0 else 0
                target_queue_index = 0

                if object_id not in self.object_id_count:
                    self.object_id_count[object_id] = 0

                if self.object_id_count[object_id] < 5:
                    # if object_id != self.previous_object_id:
                    #     print(f"object_id: {object_id}, start_line: {start_line}, end_line: {end_line}, NEW queue: {target_queue_index}")
                    # else:
                    #     print(f"object_id: {object_id}, start_line: {start_line}, end_line: {end_line}, queue: {target_queue_index}")

                    print("vehicle_result", vehicle_result)
                    print(self.vehicle_plate_result_queue_list[target_queue_index].qsize())

                    self.vehicle_plate_result_queue_list[target_queue_index].put(vehicle_result)

                    self.object_id_count[object_id] += 1
                    self.previous_object_id = object_id
                # else:
                #     print(f"object_id: {object_id} has reached the max limit of 7 occurrences.")
    

    def char_data_parsed(self, vehicle_plate_result):
        if vehicle_plate_result:
            # bg_color = last_result.get("bg_color", None)
            # plate_frame = last_result.get("frame", "")
            floor_id = vehicle_plate_result.get('floor_id', 0)
            cam_id = vehicle_plate_result.get('cam_id', "")
            # arduino_idx = last_result.get('arduino_idx')
            car_direction = vehicle_plate_result.get('car_direction')
            # start_line = last_result.get('start_line', False)
            # end_line = last_result.get('end_line', False)

            if car_direction:
                result = {
                    # "bg_color": bg_color,
                    # "frame": "",
                    "floor_id": floor_id,
                    "cam_id": cam_id,
                    # "arduino_idx": str(arduino_idx),
                    "car_direction": car_direction,
                    # "start_line": start_line,
                    # "end_line": end_line
                }
                print(json.dumps(result, indent=4))
    
    # def process_detection_result(self, result):
    #     self.vehicle_detection_result.put(result)

    def _mouse_event_debug(self, event, x, y, flags, frame):
        """Handle mouse events in debug mode."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
            print(f"Clicked coordinates: ({x}, {y})")

            normalized_points = convert_bbox_to_decimal((frame.shape[:2]), [self.clicked_points])
            print_normalized_points(normalized_points)

            draw_points_and_lines(frame, self.clicked_points)
            show_cam(f"FLOOR {self.floor_id}: {self.cam_id}", frame)

    def _mouse_event(self, event, x, y, flags, frame):
        """Handle mouse events for normal mode."""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked coordinates: ({x}, {y})")
            print(convert_bbox_to_decimal((frame.shape[:2]), [[[x, y]]]))

    def main(self):
        IS_DEBUG = True
        video_source = config.VIDEO_SOURCE_PC if IS_DEBUG else config.CAM_SOURCE_LT

        # Start processes
        self.start()

        while not self.is_model_built():
            time.sleep(0.1)
            print("Loading recognition models...")

        plat_detects = [None] * len(video_source)
        arduino_devices = [Arduino(baudrate=115200, serial_number=serial) for serial in config.SERIALS]
        frames = [None] * len(video_source)
        total_slots = {}

        # Set up detection controllers and matrix controllers
        for i in range(len(video_source)):
            idx, cam_position = check_floor(i)

            if idx not in total_slots:
                slot = self.db_floor.get_slot_by_id(idx)
                total_slots[idx] = slot["slot"]
                print(f"Slot initialized for idx {idx}: {slot}")

            arduino_text = arduino_devices[idx]
            arduino_matrix = arduino_devices[idx + 1]

            self.matrix_controller = MatrixController(arduino_matrix, max_car=18, total_car=total_slots[idx])
            self.matrix_controller.start(self.matrix_controller.get_total())

            # self.vehicle_plate_result_queue_list_2.append(mp.Queue())
            plat_detects[i] = DetectionControllerV6(arduino_text, vehicle_plate_result_queue=self.vehicle_plate_result_queue)
            plat_detects[i].start()

        while not all([m.is_model_built() for m in plat_detects]):
            time.sleep(0.1)
            print("Loading detection models...")

        # Initialize and start cameras
        caps = [CameraV1(video, is_video=True) for video in video_source]
        for cap in caps:
            print(f"Starting camera: {cap}")
            cap.start()

        try:
            print("Opening cameras...")
            while True:
                
                try:
                    for i, cap in enumerate(caps):
                        floor_id, cam_id = check_floor(i)
                        num, frames[i] = cap.read()
    
                        if frames[i] is None:
                            # print("frames[i] None", frames[i])
                            continue
    
                        # print("process frame", num)
                        plat_detects[i].process_frame({
                            "frame": frames[i].copy(),
                            "floor_id": floor_id,
                            "cam_id": cam_id
                        })
                        # frames[i] = cv2.resize(frames[i], (1080, 720))
                        height, width = frames[i].shape[:2]
    
                        slot = self.db_floor.get_slot_by_id(floor_id)
                        total_slot, vehicle_total = slot["slot"], slot["vehicle_total"]
    
                        poly_points, tracking_points, frames[i] = crop_frame(
                            frame=frames[i], height=height, width=width,
                            floor_id=floor_id, cam_id=cam_id
                        )
    
                        draw_tracking_points(frames[i], tracking_points, (height, width))
    
                        last_plate_no = self.db_vehicle_history.get_vehicle_history_by_floor_id(floor_id)["plate_no"]
                        plate_no = last_plate_no if last_plate_no else ""
    
                        add_overlay(frames[i], floor_id, cam_id, poly_points, plate_no, total_slot, vehicle_total)
    
                        # if IS_DEBUG:
                        #     draw_points_and_lines(frame, self.clicked_points)
                        #     draw_box(frame=frame, boxes=self.car_bboxes)
                        # else:
                        #     draw_box(frame=frame, boxes=self.car_bboxes)
    
                        # window_name = f"FLOOR {floor_id}: {cam_id}"
                        # show_cam(window_name, frame)
                        # cv2.setMouseCallback(
                        #     window_name,
                        #     self._mouse_event_debug if is_debug else self._mouse_event,
                        #     param=frame
                        # )
                    
                    # print("masuk frame_show")
                    frame_show = create_grid(frames, rows=2, cols=3, frame_size=640, padding=5)
                    cv2.imshow(f"Camera", frame_show)
                    # cv2.imshow(f"Camera", frames[0])
                    # print("keluar frame_show")
                except Exception as e:
                    print("Error", e)

                # Check if 'q' key is pressed to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit key 'q' pressed. Stopping...")
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            print("Terminating...")
        
        except Exception as e:
            print("Unknown error", e)

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
