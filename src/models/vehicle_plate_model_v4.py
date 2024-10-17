import os
import cv2
import numpy as np
from datetime import datetime
import torch
import threading
import queue
import multiprocessing as mp
import cv2
import numpy as np
from ultralytics import YOLO

from src.config.config import config
from src.config.logger import logger
from src.controllers.utils.util import check_background
from utils.centroid_tracking import CentroidTracker
from src.utils import get_centroids
from src.controllers.utils.util import convert_bbox_to_decimal, convert_decimal_to_bbox, point_position



class VehicleDetector:
    def __init__(self, vehicle_model, plate_model, vehicle_plate_result_queue):
        self.model = vehicle_model
        self.vehicle_plate_result_queue = vehicle_plate_result_queue
        self.centroid_tracking = CentroidTracker(maxDisappeared=75)
        self.car_direction = None
        self.prev_centroid = None
        self.num_skip_centroid = 0
        self.start_end_counter = 0

        self.frame_count = 0
        self.pd = PlateDetector(plate_model)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Received an empty image for preprocessing.")

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return image_bgr
        # return image

    def predict(self, image: np.ndarray):
        preprocessed_image = self.preprocess(image)
        results = self.model.predict(preprocessed_image, conf=0.25, device="cuda:0", verbose=False, classes=config.CLASS_NAMES)
        return results

    def get_car_image(self, frame, threshold=0.008):
        results = self.predict(frame)
        if not results[0].boxes.xyxy.cpu().tolist():
            return np.array([]), results
        boxes = results[0].boxes.xyxy.cpu().tolist()
        # print("boxes : ", boxes)
        height, width = frame.shape[:2]
        filtered_boxes = [box for box in boxes if (box[3] < height * (1 - threshold))]
        if not filtered_boxes:
            return np.array([]), results
        sorted_boxes = sorted(filtered_boxes, key=lambda x: x[3] - x[1], reverse=True)
        if len(sorted_boxes) > 0:
            box = sorted_boxes[0]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            car_frame = frame[y1:y2, x1:x2]
            if car_frame.shape[0] == 0 or car_frame.shape[1] == 0:
                return np.array([]), results
            return car_frame, results
        return np.array([]), results

    def get_centroid(self, results, line_pos):
        centro = get_centroids(results, line_pos)
        estimate_tracking, self.track_id = self.get_tracking_centroid(centro)
        if estimate_tracking != ():
            return centro
        else:
            return estimate_tracking


    def get_tracking_centroid(self, centroids):
        centroids = np.array(centroids)
        track_object = [self.centroid_tracking.update(centroid.reshape(1, -1)) for centroid in centroids][0]
        id = [i for i in track_object.keys()]
        point = [list(i.flatten()) for i in track_object.values()]
        # print("point: ", point, "id :", id)
        return point, id

    def is_car_out_v2(self, boxes):
        sorted_boxes = sorted(boxes, key=lambda x: x[3], reverse=True)
        if len(sorted_boxes) > 0:
            box0 = sorted_boxes[0]
            cx, cy = (box0[0] + box0[2]) / 2, box0[3]
            d = 200
            if self.prev_centroid is not None:
                d = abs(cy - self.prev_centroid[1])
            if d < 100:
                self.prev_centroid = cx, cy
                if len(self.centroid_sequence) > 5:
                    seq = np.array(self.centroid_sequence)
                    self.centroid_sequence = []
                    dist = seq[:-1] - seq[1:]
                    negative_indices = (dist < 0).astype(int)
                    positive_indices = (dist > 0).astype(int)
                    
                    if sum(negative_indices) > sum(positive_indices):
                        # print("mobil masuk".center(100, "="))
                        return False
                    elif sum(negative_indices) < sum(positive_indices):
                        # print("mobil keluar".center(100, "#"))
                        return True
                    else:
                        # print("mobil diam".center(100, "*"))
                        return None
                
                else:
                    self.centroid_sequence.append(cy)
            else:
                self.num_skip_centroid += 1
                if self.num_skip_centroid > 5:
                    self.prev_centroid = cx, cy
                    self.num_skip_centroid = 0
                    self.centroid_sequence = []
        return None

    def check_centroid_location(self, results, poly_points, inverse=False):
        point = (self.centroids[0][0], self.centroids[0][1])
        start = point_position(poly_points[0], poly_points[1], point, inverse=inverse)
        end = point_position(poly_points[2], poly_points[3], point, inverse=inverse)
        self.centroids = self.get_centroid(results, line_pos=False)
        if inverse:
            return end, start
        else:
            return start, end

    def processing_car_counter(self, list_data, car_direction=None):
        if car_direction is not None:
            self.car_direction = car_direction
        _, start, end = list_data

        if start and not end:
            self.passed_a = 2
        elif end:
            if self.passed_a == 2:
                self.matrix.plus_car() if not self.car_direction else self.matrix.minus_car()
            self.mobil_masuk = False
            self.passed_a = 0
        
        # logger.write(f"{self.matrix.get_total()}, {'KELUAR' if self.car_direction else 'MASUK'}, {list_data[1:-1]}".center(100, "="), logger.DEBUG)

    def save_vehicle_frame(self, vehicle_frames):
        """
        Save the vehicle frame images as image files.
        
        Args:
            vehicle_frames: List of vehicle frame images.
        """
        if not os.path.exists('vehicle_saved'):
            os.makedirs('vehicle_saved')

        for i, vehicle_frame in enumerate(vehicle_frames):
            if vehicle_frame.size > 0:
                # Create a filename with the current timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                filename = f'vehicle_saved/vehicle_frame_{timestamp}_{i}.jpg'

                # Save the vehicle frame image
                cv2.imwrite(filename, vehicle_frame)
                print(f'Saved vehicle frame: {filename}')

    def detect_vehicle(self, arduino_idx, frame: np.ndarray, floor_id: int, cam_id: str, poly_points):
        boxes = []

        if frame is None or frame.size == 0:
            print("Empty or invalid frame received.")
            return None, []

        preprocessed_image = self.preprocess(frame)
        vehicle_frame, results = self.get_car_image(preprocessed_image)

        if vehicle_frame.size > 0:
            boxes = results[0].boxes.xyxy.cpu().tolist()
            self.centroids = self.get_centroid(results, line_pos=True)
            car_direction = self.is_car_out_v2(boxes)

            if car_direction is not None or self.car_direction is None:
                self.car_direction = car_direction

            start_line, end_line = self.check_centroid_location(
                results, poly_points, inverse=self.car_direction
            )

            empty_frame = np.empty((0, 0, 3), dtype=np.uint8)

            if not start_line and not end_line:
                result = {
                    "bg_color": None,
                    "frame": empty_frame,
                    "floor_id": floor_id,
                    "cam_id": cam_id,
                    "arduino_idx": arduino_idx,
                    "car_direction": car_direction,
                    "start_line": start_line,
                    "end_line": end_line
                }
                self.vehicle_plate_result_queue.put(result)

            # Check if start and end lines are detected AND frame_count is within limit
            if vehicle_frame is not None and start_line and end_line:
                plate_results = self.pd.detect_plate(vehicle_frame)

                for plate_frame in plate_results:
                    # Process only if frame_count is still within limit
                    if self.frame_count < 3:
                        gray_plate = cv2.cvtColor(plate_frame, cv2.COLOR_BGR2GRAY)
                        bg_color = check_background(gray_plate, False)

                        self.pd.save_cropped_plate(vehicle_frame)

                        result = {
                            "bg_color": bg_color,
                            "frame": plate_frame,
                            "floor_id": floor_id,
                            "cam_id": cam_id,
                            "arduino_idx": arduino_idx,
                            "car_direction": car_direction,
                            "start_line": start_line,
                            "end_line": end_line
                        }

                        self.vehicle_plate_result_queue.put(result)
                        self.frame_count += 1

            # Reset frame_count if start_line or end_line is not both True
            if not (start_line and end_line):
                self.frame_count = 0

        return vehicle_frame, boxes


    def draw_box(self, frame, boxes):
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            color = (255, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        return frame
    
    def get_tracking_centroid(self, centroids):
        centroids = np.array(centroids)
        track_object = [self.centroid_tracking.update(centroid.reshape(1, -1)) for centroid in centroids][0]
        id = [i for i in track_object.keys()]
        point = [list(i.flatten()) for i in track_object.values()]
        # print("point: ", point, "id :", id)
        return point, id
    

    def draw_boxes(self, frame, results):
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy())
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            color = (255, 255, 255)  # White color for bounding box
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot
        
        return frame
    

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

            if cropped_plate.size > 0 and self.is_valid_cropped_plate(cropped_plate):
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