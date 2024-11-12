import threading
import queue
import multiprocessing as mp
import cv2
import numpy as np
from ultralytics import YOLO

from src.config.config import config
from src.config.logger import logger
from src.controllers.utils.util import check_background

import gc


def plate_detection(stopped, vehicle_result_queue, plate_result_queue):
    plate_model_path = config.MODEL_PATH_PLAT_v2
    plate_model = YOLO(plate_model_path)
    plate_detector = PlateDetector(plate_model)
    frame_count = 0
    prev_object_id = None

    while not stopped.is_set():
        try:
            vehicle_data = vehicle_result_queue.get()

            if vehicle_data is None:
                continue

            object_id = vehicle_data.get('object_id')
            car_frame = vehicle_data.get('frame')
            floor_id = vehicle_data.get('floor_id', 0)
            cam_id = vehicle_data.get('cam_id', "")
            arduino_idx = vehicle_data.get('arduino_idx')
            car_direction = vehicle_data.get('car_direction')
            start_line = vehicle_data.get('start_line', False)
            end_line = vehicle_data.get('end_line', False)

            print(f"cur object_id: {object_id}, prev object_id: {prev_object_id}")

            if object_id != prev_object_id:
                frame_count = 0
                prev_object_id = object_id

            if car_frame is not None:
                # print(car_frame)
                plate_results = plate_detector.detect_plate(car_frame)

                for plate in plate_results:
                    gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                    bg_color = check_background(gray_plate, False)

                    if frame_count < 10:
                        plate_detector.save_cropped_plate(plate_results)
                        result = {
                            "object_id": object_id,
                            "bg_color": bg_color,
                            "frame": plate,
                            "floor_id": floor_id,
                            "cam_id": cam_id,
                            "arduino_idx": arduino_idx,
                            "car_direction": car_direction,
                            "start_line": start_line,
                            "end_line": end_line
                        }

                        plate_result_queue.put(result) 
                        frame_count += 1  
                        # print(f"Saved and put data for object_id: {object_id}, frame_count: {frame_count}")
                    else:
                        # print(f"Skipping saving and putting data for object_id: {object_id}, frame_count: {frame_count}")
                        break 

        except Exception as e:
            print(f"Error in plate detection: {e}")



# def plate_detection(stopped, vehicle_result_queue, plate_result_queue):
#     plate_model_path = config.MODEL_PATH_PLAT_v2
#     plate_model = YOLO(plate_model_path)
#     plate_detector = PlateDetector(plate_model)
#     frame_count = 0

#     while not stopped.is_set():
#         try:
#             vehicle_data = vehicle_result_queue.get()

#             if vehicle_data is None:
#                 continue

#             object_id = vehicle_data.get('object_id')
#             car_frame = vehicle_data.get('frame')
#             floor_id = vehicle_data.get('floor_id', 0)
#             cam_id = vehicle_data.get('cam_id', "")
#             arduino_idx = vehicle_data.get('arduino_idx')
#             car_direction = vehicle_data.get('car_direction')
#             start_line = vehicle_data.get('start_line', False)
#             end_line = vehicle_data.get('end_line', False)

#             # Create an empty frame to use in results
#             empty_frame = np.empty((0, 0, 3), dtype=np.uint8)

#             # Check if both start_line and end_line are False
#             if not start_line and not end_line:
#                 # Reset frame_count if both lines are False
#                 frame_count = 0
#                 result = {
#                     "object_id": object_id,
#                     "bg_color": None,
#                     "frame": empty_frame,
#                     "floor_id": floor_id,
#                     "cam_id": cam_id,
#                     "arduino_idx": arduino_idx,
#                     "car_direction": car_direction,
#                     "start_line": start_line,
#                     "end_line": end_line
#                 }
#                 plate_result_queue.put(result)
#                 continue

#             if car_frame is not None and start_line and end_line:
#                 plate_results = plate_detector.detect_plate(car_frame)

#                 for plate in plate_results:
#                     gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
#                     bg_color = check_background(gray_plate, False)

#                     if frame_count < 5:
#                         plate_detector.save_cropped_plate(plate_results)
#                         result = {
#                             "object_id": object_id,
#                             "bg_color": bg_color,
#                             "frame": plate,
#                             "floor_id": floor_id,
#                             "cam_id": cam_id,
#                             "arduino_idx": arduino_idx,
#                             "car_direction": car_direction,
#                             "start_line": start_line,
#                             "end_line": end_line
#                         }

#                         plate_result_queue.put(result)
#                         frame_count += 1

#             if not start_line or not end_line:
#                 frame_count = 0
            
#             del vehicle_data
#             gc.enable()
#             gc.collect()
#             gc.disable()

#         except Exception as e:
#             print(f"Error in plate detection: {e}")

class PlateDetector:
    def __init__(self, model):
        self.model = model

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return image_bgr

    def predict(self, image: np.ndarray):
        preprocessed_image = self.preprocess(image)
        results = self.model.predict(preprocessed_image, conf=0.3, device="cuda:0", verbose=False)
        return results

    def detect_plate(self, frame, is_save=False):
        results = self.predict(frame)

        if not results:
            print("[PlateDetector] No plates detected.")
            return []

        bounding_boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if results else []
        if not bounding_boxes:
            return []

        cropped_plates = self.get_cropped_plates(frame, bounding_boxes)
        # print("cropped_plates: ", cropped_plates)

        if is_save:
            self.save_cropped_plate(cropped_plates)

        return cropped_plates

    def save_cropped_plate(self, cropped_plates):
        """
        Save the cropped plate regions as image files.
        Args:
            cropped_plates: List of cropped plate images.
        """
        import os
        from datetime import datetime

        if not os.path.exists('plate_saved'):
            os.makedirs('plate_saved')

        for i, cropped_plate in enumerate(cropped_plates):
            if cropped_plate.size > 0:
                # Create a filename with the current timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                filename = f'plate_saved/{timestamp}.jpg'

                # Save the cropped plate image
                cv2.imwrite(filename, cropped_plate)

    def is_valid_cropped_plate(self, cropped_plate):
        """Check if the cropped plate meets the size requirements."""
        height, width = cropped_plate.shape[:2]
        if height < 55 or width < 100:
            return False
        if height >= width:
            return False
        compare = abs(height - width)
        if compare <= 100 or compare >= 400:
            return False
        return True

    def get_cropped_plates(self, frame, boxes):
        """
        Extract cropped plate images based on bounding boxes.
        Args:
            frame: The original image/frame.
            boxes: List of bounding boxes (each box is [x1, y1, x2, y2]).

        Returns:
            cropped_plates: List of cropped plate images.
        """
        height, width, _ = frame.shape
        cropped_plates = []

        for box in boxes:
            x1, y1, x2, y2 = [max(0, min(int(coord), width if i % 2 == 0 else height)) for i, coord in enumerate(box)]
            cropped_plate = frame[y1:y2, x1:x2]

            # if cropped_plate.size > 0 and self.is_valid_cropped_plate(cropped_plate):
            if cropped_plate.size > 0:
                cropped_plates.append(cropped_plate)

        return cropped_plates


    def draw_boxes(self, frame, boxes):
        """
        Draw bounding boxes for detected plates on the frame.
        Args:
            frame: The original image/frame.
            boxes: List of bounding boxes to draw (each box is [x1, y1, x2, y2]).
        """
        height, width, _ = frame.shape

        for box in boxes:
            x1, y1, x2, y2 = [max(0, min(int(coord), width if i % 2 == 0 else height)) for i, coord in enumerate(box)]

            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red

        return frame