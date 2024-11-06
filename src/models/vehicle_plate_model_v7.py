import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from shapely.geometry import Polygon
import csv
import pandas as pd
import time 

from src.config.config import config

from utils.centroid_tracking import CentroidTracker
from src.controllers.utils.util import (
    check_background, 
    point_position,
    convert_normalized_to_pixel,
    convert_normalized_to_pixel_lines,
    convert_bbox_to_decimal,
    is_point_in_polygon,
    get_centroid
)

from src.controllers.utils.display import (
    print_normalized_points,
    draw_points_and_lines,
    show_cam
)



class VehicleDetector:
    def __init__(self, model_path, is_vehicle_model):
        self.is_vehicle_model = is_vehicle_model
        self.centroid_tracking = CentroidTracker(maxDisappeared=75)
        self.model = model_path
        self.car_direction = None
        self.prev_centroid = None
        self.num_skip_centroid = 0
        self.tracking_points = []
        self.poly_points = []
        self.poly_bbox = []
        self.centroids = None
        self.clicked_points = []
        self.car_bboxes = []
        self.plate_info = []
        self.arduino_idx = 0
        self.max_num_frame = 1
        self.num_add_plate_size = 5
        self.frame_count = 0
        self.prev_object_id = None
        self.prev_floor_id = None
        self.prev_cam_id = None
        self.frame_count_per_object = {}
        self.class_index = [2, 7, 5] if self.is_vehicle_model else [0, 1, 2]
        self.prev_centroids = {}
        self.movement_count = {}


    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Received an empty image for preprocessing.")

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return image_bgr

    def save_plate(self, frame, plate_box, is_save=True, output_folder="output_plate"):
        """Save the detected plate without bounding box to disk."""
        if is_save:
            os.makedirs(output_folder, exist_ok=True)

            timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S%f')[:-3]
            plate_x1, plate_y1, plate_x2, plate_y2 = map(int, plate_box)
            plate_image = frame[plate_y1:plate_y2, plate_x1:plate_x2].copy()

            file_name = f"{output_folder}/plate_{timestamp}.jpg"
            cv2.imwrite(file_name, plate_image)
            print(f"Plate saved as: {file_name}")

    def get_centroid_object(self, bbox):
        """Calculate the centroid of a bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        centroid_x = (x1 + x2) // 2
        centroid_y = (y1 + y2) // 2
        return (centroid_x, centroid_y)

    def _mouse_event_debug(self, event, x, y, flags, frame):
        """Handle mouse events in debug mode."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
            print(f"Clicked coordinates: ({x}, {y})")

            normalized_points = convert_bbox_to_decimal((frame.shape[:2]), [self.clicked_points])
            print_normalized_points(normalized_points)

            draw_points_and_lines(frame, self.clicked_points)
            show_cam(f"FLOOR {self.floor_id}: {self.cam_id}", frame)

    def is_valid_cropped_plate(self, cropped_plate, floor_id, cam_id):
        """Check if the cropped plate meets the size requirements and save dimensions to a CSV file."""
        height, width = cropped_plate.shape[:2]
        # print(f'height: {height} & width: {width}')

        # Save height and width to CSV
        # self.save_dimensions_to_csv(height, width)
        # self.save_dimensions_to_excel(height, width)

        if floor_id == 2:
            if cam_id == "IN":
                # if height < 30 or width < 105:
                #     return False
                if height >= width:
                    return False
                # compare = abs(height - width)
                # if compare <= 35 or compare >= 80:
                #     return False
            else:
                # if height < 35 or width < 75 or width >= 200:
                #     return False
                if height >= width:
                    return False
                # compare = abs(height - width)
                # if compare <= 35 or compare >= 80:
                #     return False

        elif floor_id == 3:
            if cam_id == "IN":
                # if height < 30 or width < 90:
                #     return False
                if height >= width:
                    return False
                # compare = abs(height - width)
                # if compare <= 35 or compare >= 80:
                #     return False
            else:
                # if height < 30 or width < 90 or width >= 150:
                #     return False
                if height >= width:
                    return False
                # compare = abs(height - width)
                # if compare <= 35 or compare >= 85: # 120:
                #     return False
        else:
            # if height < 30 or width < 90:
            #     return False
            if height >= width:
                return False
            # compare = abs(height - width)
            # if compare <= 35 or compare >= 80:
            #     return False

        # if height < 55 or width < 100:
        #     return False
        # if height >= width:
        #     return False
        # compare = abs(height - width)
        # if compare <= 110 or compare >= 400:
        #     return Falseq
        return True

    def save_dimensions_to_excel(self, height, width, floor_id, cam_id):
        """Save height and width to an Excel file named with the current date."""
        current_date = datetime.now()
        current_date_format = current_date.strftime("%Y_%m_%d") 
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
   
        filename = current_date_format + ".xlsx"

        new_data = pd.DataFrame([[height, width, floor_id, cam_id, timestamp]], columns=['height', 'width', 'floor_id', 'cam_id', 'created_date'])

        if os.path.isfile(filename):
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                new_data.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        else:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                new_data.to_excel(writer, index=False, sheet_name='Sheet1')

    def save_dimensions_to_csv(self, height, width):
        """Save height and width to a CSV file named with the current date."""
        current_date = datetime.now()
        filename = current_date.strftime("%Y_%m_%d") + ".csv"

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)

            file.seek(0, 2)
            if file.tell() == 0:
                writer.writerow(['Height', 'Width'])
            writer.writerow([height, width])

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
                timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                filename = f'plate_saved/{timestamp}.jpg'

                cv2.imwrite(filename, cropped_plate)

    def is_car_out(self, boxes):
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

    def get_tracking_centroid(self, centroids):
        if not centroids:
            return [], []

        centroids = np.array(centroids)
        track_object = [self.centroid_tracking.update(centroid.reshape(1, -1)) for centroid in centroids]
        ids = [i for i in track_object[0].keys()]
        points = [list(i.flatten()) for i in track_object[0].values()]
        return points, ids

    def check_centroid_location(self, results, poly_points, inverse=False):
        if not self.centroids:
            return [], []

        point = (self.centroids[0][0], self.centroids[0][1])
        start = point_position(poly_points[0], poly_points[1], point, inverse=inverse)
        end = point_position(poly_points[2], poly_points[3], point, inverse=inverse)
        self.centroids = get_centroid(results, line_pos=False)
        if inverse:
            return end, start
        else:
            return start, end

    def crop_frame_with_polygon(self, frame, poly_points):
        """Crop the frame to the area inside the polygon."""

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(poly_points, dtype=np.int32)], 255)
        cropped_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        x, y, w, h = cv2.boundingRect(np.array(poly_points))
        cropped_frame = cropped_frame[y:y+h, x:x+w]
        
        return cropped_frame

    def draw_polygon(self, frame, polygon_points, color=(255, 0, 0)):
        """Draw the polygon defined by polygon_points on the frame."""
        polygon_points = np.array(polygon_points, dtype=np.int32)
        cv2.polylines(frame, [polygon_points], isClosed=True, color=color, thickness=2)

    # def check_car_touch_line(self, frame_size, car_info, polygon_area):
    #     """
    #     Cek apakah ada centroid objek yang masuk ke dalam area poligon.
    #     """
    #     polygon_points = [
    #         convert_normalized_to_pixel_lines(point, frame_size)
    #         for point in polygon_area
    #     ]

    #     for (object_id, confidence, bbox, class_name) in car_info:
    #         car_id = object_id
    #         centroid = self.get_centroid_object(bbox)

    #         if is_point_in_polygon(centroid, polygon_points):
    #             return True

    #     return False

    def check_car_touch_line(self, frame_size, car_info, polygon_area, cam_id):
        polygon_points = [
            convert_normalized_to_pixel_lines(point, frame_size)
            for point in polygon_area
        ]

        for (object_id, confidence, bbox, class_name) in car_info:
            car_id = object_id
            centroid = self.get_centroid_object(bbox)

            if is_point_in_polygon(centroid, polygon_points):
                if car_id in self.prev_centroids:
                    prev_centroid = self.prev_centroids[car_id]

                    if centroid[1] > prev_centroid[1]:  # Moving downwards (top to bottom)
                        if cam_id == "IN":
                            # print(f"Object ID {car_id} is moving Downwards (Top to Bottom), cam_id: IN")
                            if car_id not in self.movement_count:
                                self.movement_count[car_id] = 0
                            self.movement_count[car_id] += 1
                            if self.movement_count[car_id] >= 5:
                                self.prev_centroids[car_id] = centroid
                                return True, False
                        elif cam_id == "OUT":
                            # print(f"Object ID {car_id} is moving Downwards (Top to Bottom), cam_id: OUT")
                            self.prev_centroids[car_id] = centroid
                            return True, True

                    elif centroid[1] < prev_centroid[1]:  # Moving upwards (bottom to top)
                        if cam_id == "IN":
                            # print(f"Object ID {car_id} is moving Upwards (Bottom to Top), cam_id: IN")
                            self.movement_count[car_id] = 0
                            self.prev_centroids[car_id] = centroid
                            return True, True
                        elif cam_id == "OUT":
                            # print(f"Object ID {car_id} is moving Upwards (Bottom to Top), cam_id: OUT")
                            if car_id not in self.movement_count:
                                self.movement_count[car_id] = 0
                            self.movement_count[car_id] += 1
                            if self.movement_count[car_id] >= 5:
                                self.prev_centroids[car_id] = centroid
                                return True, False

                else:
                    self.prev_centroids[car_id] = centroid
                    # print(f"Object ID {car_id} just entered the polygon area.")
            
            else:
                return False, False

        return False, None

    def process_car(self, results):
        """Process the detection of cars and return car information."""
        car_boxes = []
        car_info = []

        self.class_names = ['car', 'bus', 'truck'] if self.is_vehicle_model else ['car', 'plate', 'truck']

        for r in results[0].boxes:
            if r.id is not None and r.cls is not None and r.conf is not None:
                class_id = int(r.cls.item())
                object_id = int(r.id.item())
                bbox = r.xyxy[0].cpu().numpy().tolist()
                confidence = float(r.conf.item())

                # print("class_id: ", class_id)
                # if class_id == 0:
                #     print(f'object_id: {object_id}')

                class_name = self.class_names[class_id]
                
                if self.is_vehicle_model:
                    if class_name in ["car", "bus", "truck"]:
                        car_boxes.append(bbox)
                        car_info.append((object_id, confidence, bbox, str(class_id)))
                else:
                    if class_name == "car":
                        # print("CAR_FRAME DETECTED")
                        car_boxes.append(bbox)
                        car_info.append((object_id, confidence, bbox, str(class_id)))

        return car_boxes, car_info

    def process_plate(self, results, original_frame, car_boxes, floor_id, cam_id, is_save=True):
        """Process the detection of plates within the original frame size and return plate info."""
        plate_info = []
        cropped_plates = []

        bounding_boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if results else []
        if not bounding_boxes:
            return plate_info, cropped_plates

        for r in results[0].boxes:
            if r.id is not None and r.cls is not None and r.conf is not None:
                class_id = int(r.cls.item())
                class_name = self.class_names[class_id]

                if class_name == "plate":
                    plate_bbox = r.xyxy[0].cpu().numpy().tolist()
                    confidence = float(r.conf.item())
                    plate_x1, plate_y1, plate_x2, plate_y2 = map(int, plate_bbox)

                    for car_box in car_boxes:
                        car_x1, car_y1, car_x2, car_y2 = map(int, car_box)

                        if (
                            car_x1 <= plate_x1 <= car_x2 and car_y1 <= plate_y1 <= car_y2 and
                            car_x1 <= plate_x2 <= car_x2 and car_y1 <= plate_y2 <= car_y2
                        ):
                            plate_info.append((r.id.item(), confidence, plate_bbox, class_name))
                            
                            plate_y1 = max(plate_y1 - self.num_add_plate_size, 0)
                            plate_y2 = min(plate_y2 + self.num_add_plate_size, original_frame.shape[0])
                            plate_x1 = max(plate_x1 - self.num_add_plate_size, 0)
                            plate_x2 = min(plate_x2 + self.num_add_plate_size, original_frame.shape[1])

                            cropped_plate = original_frame[plate_y1:plate_y2, plate_x1:plate_x2]
                            if cropped_plate.size > 0: 
                            # and self.is_valid_cropped_plate(cropped_plate, floor_id, cam_id): 
                                # Save height and width to CSV
                                # self.save_dimensions_to_csv(height, width)

                                height, width = cropped_plate.shape[:2]
                                # print(f'height: {height} & width: {width}')
                                self.save_dimensions_to_excel(height, width, floor_id, cam_id)                                
                                cropped_plates.append(cropped_plate)
                                # self.save_cropped_plate([cropped_plate])

                            return plate_info, cropped_plates

        return plate_info, cropped_plates

    def check_blur(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        return laplacian_var

    def vehicle_detect(self, arduino_idx, frame, floor_id, cam_id, tracking_points, poly_bbox):
        height, width = frame.shape[:2]
        frame_size = (width, height)

        area_selection = convert_normalized_to_pixel(tracking_points, (height, width))
        cropped_frame = self.crop_frame_with_polygon(frame, area_selection)
        cropped_frame_copy = cropped_frame.copy()

        preprocessed_image = self.preprocess(cropped_frame)

        results = self.model.track(preprocessed_image, conf=0.25, persist=True, verbose=False)

        self.car_bboxes, car_info = self.process_car(results)

        for car_bbox in self.car_bboxes:
            self.centroids = self.get_centroid_object(car_bbox)

        # direction = self.is_car_out(self.car_bboxes)
        # if direction is not None or self.car_direction is None:
        #     self.car_direction = direction

        self.object_id = car_info[0][0] if car_info else None

        is_centroid_inside, direction = self.check_car_touch_line(frame_size, car_info, poly_bbox, cam_id)

        if direction is not None or self.car_direction is None:
            self.car_direction = direction

        # if direction is True:
        #     print("Car is entering the area from Top to Bottom (Downwards)")
        # elif direction is False:
        #     print("Car is entering the area from Bottom to Top (Upwards)")

        if is_centroid_inside and not self.is_vehicle_model:
            self.plate_info, plate_frames = self.process_plate(results, cropped_frame_copy, self.car_bboxes, floor_id, cam_id)

            if self.plate_info:
                self.plate_bbox = self.plate_info[0][2]
            else:
                self.plate_bbox = None

            # Initialize `frame_count_per_object` and previous object check
            if self.object_id not in self.frame_count_per_object:
                self.frame_count_per_object[self.object_id] = 0

            # Check if the current floor and cam match the previous object
            if (self.object_id != self.prev_object_id or
                (self.prev_floor_id != floor_id or self.prev_cam_id != cam_id)):
                self.frame_count_per_object[self.object_id] = 0

            if self.frame_count_per_object[self.object_id] < self.max_num_frame:
                for plate_frame in plate_frames:
                    check_blur_img = self.check_blur(plate_frame)
                    print("BLUR IMAGE VALUE: ", check_blur_img)
                    if check_blur_img == 1000:
                        continue

                    print("PLATE_FRAME DETECTED")
                    self.save_cropped_plate([plate_frame])

                    gray_plate = cv2.cvtColor(plate_frame, cv2.COLOR_BGR2GRAY)
                    bg_color = check_background(gray_plate, False)

                    vehicle_plate_data = {
                        "object_id": self.object_id,
                        "bbox": self.plate_bbox,
                        "is_centroid_inside": is_centroid_inside,
                        "bg_color": bg_color,
                        "frame": plate_frame,
                        "floor_id": floor_id,
                        "cam_id": cam_id,
                        "arduino_idx": arduino_idx,
                        "car_direction": self.car_direction,
                        "start_line": True,
                        "end_line": True
                    }

                    self.frame_count_per_object[self.object_id] += 1
                    self.prev_object_id = self.object_id
                    self.prev_floor_id = floor_id
                    self.prev_cam_id = cam_id  # Store the current floor and cam as the previous

                    return vehicle_plate_data, cropped_frame_copy, is_centroid_inside, car_info

            else:
                print(f"Skipping saving for object_id: {self.object_id}, frame_count: {self.frame_count_per_object[self.object_id]}")

        return {}, cropped_frame_copy, is_centroid_inside, car_info


    # def vehicle_detect(self, arduino_idx, frame, floor_id, cam_id, tracking_points, poly_bbox):
    #     height, width = frame.shape[:2]
    #     frame_size = (width, height)

    #     area_selection = convert_normalized_to_pixel(tracking_points, (height, width))
    #     cropped_frame = self.crop_frame_with_polygon(frame, area_selection)
    #     cropped_frame_copy = cropped_frame.copy()

    #     preprocessed_image = self.preprocess(cropped_frame)

    #     results = self.model.track(preprocessed_image, conf=0.25, persist=True, verbose=False)

    #     self.car_bboxes, car_info = self.process_car(results)

    #     for car_bbox in self.car_bboxes:
    #         self.centroids = self.get_centroid_object(car_bbox)

    #     direction = self.is_car_out(self.car_bboxes)
    #     if direction is not None or self.car_direction is None:
    #         self.car_direction = direction

    #     # for (object_id, confidence, bbox, class_name) in car_info:
    #     #     self.object_id = object_id
        
    #     self.object_id = car_info[0][0] if car_info else None

    #     is_centroid_inside = self.check_car_touch_line(frame_size, car_info, poly_bbox)

    #     if is_centroid_inside and not self.is_vehicle_model:
    #         self.plate_info, plate_frames = self.process_plate(results, cropped_frame_copy, self.car_bboxes, floor_id, cam_id)

    #         # for (object_id, confidence, bbox, class_name) in self.plate_info:
    #         #     self.plate_bbox = bbox

    #         if self.plate_info:
    #             self.plate_bbox = self.plate_info[0][2]
    #         else:
    #             self.plate_bbox = None

    #         if self.object_id != self.prev_object_id:
    #             self.frame_count_per_object[self.object_id] = 0

    #         if self.object_id not in self.frame_count_per_object:
    #             self.frame_count_per_object[self.object_id] = 0

    #         if self.frame_count_per_object[self.object_id] < self.max_num_frame:
    #             for plate_frame in plate_frames:
    #                 print("PLATE_FRAME DETECTED")
    #                 self.save_cropped_plate([plate_frame])

    #                 gray_plate = cv2.cvtColor(plate_frame, cv2.COLOR_BGR2GRAY)
    #                 bg_color = check_background(gray_plate, False)

    #                 # print("bg_color: ", bg_color)

    #                 vehicle_plate_data = {
    #                     "object_id": self.object_id,
    #                     "bbox": self.plate_bbox,
    #                     "is_centroid_inside": is_centroid_inside,
    #                     "bg_color": bg_color,
    #                     "frame": plate_frame,
    #                     "floor_id": floor_id,
    #                     "cam_id": cam_id,
    #                     "arduino_idx": arduino_idx,
    #                     "car_direction": self.car_direction,
    #                     "start_line": True,
    #                     "end_line": True
    #                 }

    #                 self.frame_count_per_object[self.object_id] += 1
    #                 self.prev_object_id = self.object_id

    #                 return vehicle_plate_data, cropped_frame_copy, is_centroid_inside, car_info

    #         else:
    #             print(f"Skipping saving for object_id: {self.object_id}, frame_count: {self.frame_count_per_object[self.object_id]}")

    #     return {}, cropped_frame_copy, is_centroid_inside, car_info