import os
import cv2
import numpy as np
from datetime import datetime
import torch

from src.config.config import config
from src.config.logger import logger
from utils.centroid_tracking import CentroidTracker
from src.utils import get_centroids
from src.controllers.utils.util import convert_bbox_to_decimal, convert_decimal_to_bbox, point_position



class VehicleDetector:
    def __init__(self, model, vehicle_result_queue):
        self.model = model
        self.vehicle_result_queue = vehicle_result_queue
        self.centroid_tracking = CentroidTracker(maxDisappeared=75)
        self.car_direction = None
        self.prev_centroid = None
        self.num_skip_centroid = 0
        self.passed_a = 0
        self.mobil_masuk = False
        self.start_end_counter = 0

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Received an empty image for preprocessing.")

        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        # return image_bgr
        return image

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

    def detect_vehicle(self, arduino_idx, frame: np.ndarray, floor_id: int, cam_id: str, matrix, poly_points):
        self.matrix = matrix
        boxes = []

        if frame is None or frame.size == 0:
            print("Empty or invalid frame received.")
            return None, []

        preprocessed_image = self.preprocess(frame)
        vehicle_frame, results = self.get_car_image(preprocessed_image)

        if vehicle_frame.size > 0:
            # self.save_vehicle_frame(vehicle_frames=vehicle_frame)
            boxes = results[0].boxes.xyxy.cpu().tolist()
            self.centroids = self.get_centroid(results, line_pos=True)
            direction = self.is_car_out_v2(boxes)

            if direction is not None or self.car_direction is None:
                self.car_direction = direction

            start_line, end_line = self.check_centroid_location(
                results, poly_points, inverse=self.car_direction
            )

            if not start_line and not end_line:
                vehicle_frame = np.empty((0, 0, 3), dtype=np.uint8)

            # print(f'VEHICLE : start_line: {start_line}, End_line: {end_line}')

            self.mobil_masuk = vehicle_frame.size > 0

            vehicle_data = {
                'arduino_idx': arduino_idx,
                'frame': vehicle_frame,
                'floor_id': floor_id,
                'cam_id': cam_id,
                'car_direction': self.car_direction,  # True or False
                'mobil_masuk': self.mobil_masuk,  # True or False
                'start_line': start_line,  # Boolean start flag
                'end_line': end_line  # Boolean end flag
            }

            self.vehicle_result_queue.put(vehicle_data)

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