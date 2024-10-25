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
from src.Integration.service_v1.controller.plat_controller import PlatController
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.fetch_api_controller import FetchAPIController
from src.Integration.service_v1.controller.vehicle_history_controller import VehicleHistoryController
from utils.centroid_tracking import CentroidTracker
from src.controllers.utils.util import check_background

def show_cam(text, image, max_width=1080, max_height=720):
    res_img = resize_image(image, max_width, max_height)
    cv2.imshow(text, res_img)
    # cv2.imshow(text, image)

def show_text(text, image, x_shape, y_shape, color=(255, 255, 255)):
    image = cv2.rectangle(image, (x_shape - 10, y_shape + 20), (x_shape + 400, y_shape - 40), (0, 0, 0), -1)
    return cv2.putText(image, text, (x_shape, y_shape), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scale = max_width / width if width > height else max_height / height
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return image

def show_box(image, box, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

def show_polygon(image, points: list[list[int]], color=(0, 255, 0)):
    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
    return image

def show_line(frame, point1, point2):
    cv2.line(frame, point1, point2, (0, 255, 0), 2)
    return frame

def convert_bbox_to_decimal(img_dims, polygons):
    height, width = img_dims
    normalized_polygons = []

    for bbox in polygons:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 2:  # Koordinat titik (x, y)
            normalized_bbox = (bbox[0] / width, bbox[1] / height)
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:  # Bounding box (x, y, w, h)
            normalized_bbox = (
                bbox[0] / width, bbox[1] / height,
                bbox[2] / width, bbox[3] / height
            )
        else:
            raise ValueError(f"Unexpected bbox format: {bbox}")
        
        normalized_polygons.append(normalized_bbox)

    return normalized_polygons


# def convert_bbox_to_decimal(img_dims, polygons):
#     height, width = img_dims
#     # normalized_polygons = [
#     #     # ((bbox[0] / width, bbox[1] / height), (bbox[2] / width, bbox[3] / height)) for bbox in polygons
#     #     (bbox[0] / width, bbox[1] / height) for bbox in polygons
#     # ]

#     normalized_polygons = []
#     for bbox in polygons:
#         if isinstance(bbox, list):  # If it's a list of tuples
#             normalized_bbox = [(pt[0] / width, pt[1] / height) for pt in bbox]
#         else:  # If it's a simple tuple (x, y, w, h)
#             normalized_bbox = (bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height)
#         normalized_polygons.append(normalized_bbox)

#     return normalized_polygons

def print_normalized_points(normalized_points):
    print("[")
    for point in normalized_points:
        if isinstance(point[0], tuple):  # If `point` is a list of tuples (a polygon)
            for sub_point in point:
                print(f"    ({sub_point[0]:.16f}, {sub_point[1]:.16f}),")
        else:  # If `point` is a flat tuple (x, y)
            print(f"    ({point[0]:.16f}, {point[1]:.16f}),")
    print("]")


# def print_normalized_points(normalized_points):
#     print("[")
#     for point in normalized_points:
#         print(f"    ({point[0]:.16f}, {point[1]:.16f}),")
#     print("]")

def convert_normalized_to_pixel(points, img_dims):
    height, width = img_dims
    pixel_points = [(int(x * width), int(y * height)) for (x, y) in points]
    return pixel_points

def draw_tracking_points(frame, points, img_dims):
    pixel_points = convert_normalized_to_pixel(points, img_dims)

    for i in range(len(pixel_points)):
        start_point = pixel_points[i]
        end_point = pixel_points[(i + 1) % len(pixel_points)]
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    for point in pixel_points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

def show_yolo(frame, object_info, color, draw_centroid=False):
    """Display YOLO-detected objects (either cars or plates) on the frame."""
    for (object_id, confidence, bbox, class_name) in object_info:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox)

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
        # Display the object details (ID, confidence)
        # cv2.putText(frame, f"{class_name} ID: {object_id}, Conf: {confidence:.2f}", 
        #             (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
        #             0.5, color, 1)

        # Optionally draw the centroid
        if draw_centroid:
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
            # cv2.putText(frame, "Centroid", (centroid_x - 20, centroid_y - 10), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

def add_overlay(frame, floor_id, cam_id, poly_points, plate_no, total_slot, vehicle_total):
    show_text(f"Floor : {floor_id} {cam_id}", frame, 5, 50)
    show_text(f"Plate No. : {plate_no}", frame, 5, 100)
    color = (0, 255, 0) if total_slot > 0 else (0, 0, 255)
    show_text(f"P. Spaces Available : {total_slot}", frame, 5, 150, color)
    show_text(f"Car Total : {vehicle_total}", frame, 5, 200)

    show_line(frame, poly_points[0], poly_points[1])
    show_line(frame, poly_points[2], poly_points[3])

# def draw_points_and_lines(frame, clicked_points):
#     for point in clicked_points:
#         cv2.circle(frame, point, 5, (0, 0, 255), -1)
#     if len(clicked_points) > 1:
#         for i in range(len(clicked_points)):
#             start_point = clicked_points[i]
#             end_point = clicked_points[(i + 1) % len(clicked_points)]
#             cv2.line(frame, start_point, end_point, (255, 0, 0), 1)



def draw_box(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        color = (255, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
    return frame

def show_object(frame, object_id, bbox, class_name, confidence, color, draw_centroid=False):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{class_name} ID: {object_id}, Conf: {confidence:.2f}", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if draw_centroid:
        centroid_x = (x1 + x2) // 2
        centroid_y = (y1 + y2) // 2
        cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
        cv2.putText(frame, "Centroid", (centroid_x - 20, centroid_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def convert_decimal_to_bbox(img_dims, polygons):
    polygons_np = np.array(polygons)
    height, width = img_dims
    size = np.array([width, height], dtype=float)
    polygons_np *= size
    # height, width = img_dims
    # for i, pol in enumerate(polygons):
    #     for index, bbox in enumerate(pol):
    #         polygons[i][index] = (int(bbox[0] * width), int(bbox[1] * height))
    return polygons_np.round().astype(int)

def define_tracking_polygon(height, width, floor_id, cam_id):
    if floor_id == 2:
        tracking_point = config.TRACKING_POINT2_F1_IN if cam_id == "IN" else config.TRACKING_POINT2_F1_OUT
        polygon_point = config.POLYGON_POINT_LT2_IN if cam_id == "IN" else config.POLYGON_POINT_LT2_OUT
        POLY_BOX = config.POLY_BBOX_F2_IN if cam_id == "IN" else config.POLY_BBOX_F2_OUT
    elif floor_id == 3:
        tracking_point = config.TRACKING_POINT2_F3_IN if cam_id == "IN" else config.TRACKING_POINT2_F3_OUT
        polygon_point = config.POLYGON_POINT_LT3_IN if cam_id == "IN" else config.POLYGON_POINT_LT3_OUT
        POLY_BOX = config.POLY_BBOX_F3_IN if cam_id == "IN" else config.POLY_BBOX_F3_OUT
    elif floor_id == 4:
        tracking_point = config.TRACKING_POINT2_F4_IN if cam_id == "IN" else config.TRACKING_POINT2_F4_OUT
        polygon_point = config.POLYGON_POINT_LT4_IN if cam_id == "IN" else config.POLYGON_POINT_LT4_OUT
        POLY_BOX = config.POLY_BBOX_F4_IN if cam_id == "IN" else config.POLY_BBOX_F4_OUT
    elif floor_id == 5:
        tracking_point = config.TRACKING_POINT2_F5_IN if cam_id == "IN" else config.TRACKING_POINT2_F5_OUT
        polygon_point = config.POLYGON_POINT_LT5_IN if cam_id == "IN" else config.POLYGON_POINT_LT5_OUT
        POLY_BOX = config.POLY_BBOX_F5_IN if cam_id == "IN" else config.POLY_BBOX_F5_OUT
    else:
        return [], []

    poly_points = convert_decimal_to_bbox((height, width), polygon_point)
    
    return poly_points, tracking_point, POLY_BOX

def point_position(line1, line2, point, inverse=False):
    x1, y1 = line1
    x2, y2 = line2
    px, py = point
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    d = A * px + B * py + C
    if d > 0:
        return False if not inverse else True
    elif d < 0:
        return True if not inverse else False

class VehicleDetector:
    def __init__(self, model_path, video_path, floor_id, cam_id, is_vehicle_model, output_folder="output"):
        self.floor_id = floor_id
        self.cam_id = cam_id
        self.is_vehicle_model = is_vehicle_model
        self.centroid_tracking = CentroidTracker(maxDisappeared=75)
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_folder = output_folder

        self.car_direction = None
        self.prev_centroid = None
        self.num_skip_centroid = 0
        self.previous_centroids = {}
        self.object_status = {}

        self.tracking_points = []
        self.poly_points = []
        self.poly_bbox = []
        self.centroids = None
        self.clicked_points = []
        self.car_bboxes = []
        self.plate_info = []
        self.arduino_idx = 0
        self.max_num_frame = 0
        self.frame_count = 0
        self.prev_object_id = None
        self.frame_count_per_object = {}

        self.db_plate = PlatController()
        self.db_floor = FloorController()
        self.db_mysn = FetchAPIController()
        self.db_vehicle_history = VehicleHistoryController()
        self.frame_size = None
        os.makedirs(self.output_folder, exist_ok=True)

        # self.class_names = ['car', 'bus', 'truck'] if self.is_vehicle_model else ['car', 'plate', 'truck']
        self.class_index = [2, 7, 5] if self.is_vehicle_model else [0, 1, 2]
        # self.class_indices = self.get_class_indices()

        self.video_path = video_path  # Ensure video path is stored in the class

    # def get_class_indices(self):
    #     """Map class names to indices based on the model's available classes."""
    #     available_classes = self.model.names  # Get class names from the YOLO model
    #     class_indices = [i for i, name in available_classes.items() if name in self.class_names]
    #     return class_indices

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Received an empty image for preprocessing.")

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return image_bgr


    def draw_points_and_lines(self, frame):
        """Draw points and lines on the frame."""
        for point in self.clicked_points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)  # Draw red points
        if len(self.clicked_points) > 1:
            for i in range(len(self.clicked_points)):
                start_point = self.clicked_points[i]
                end_point = self.clicked_points[(i + 1) % len(self.clicked_points)]
                cv2.line(frame, start_point, end_point, (255, 0, 0), 1)  # Draw blue lines

    def save_plate(self, frame, plate_box, is_save=True):
        """Save the detected plate without bounding box to disk."""
        if is_save:
            # Get the current timestamp for the file name
            timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S%f')[:-3]

            # Extract plate region from the original frame using bounding box coordinates
            plate_x1, plate_y1, plate_x2, plate_y2 = map(int, plate_box)
            plate_image = frame[plate_y1:plate_y2, plate_x1:plate_x2].copy()

            # Save the image with the timestamp
            file_name = f"{self.output_folder}/plate_{timestamp}.jpg"
            cv2.imwrite(file_name, plate_image)
            print(f"Plate saved as: {file_name}")

    def get_centroid_object(self, bbox):
        """Calculate the centroid of a bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        centroid_x = (x1 + x2) // 2
        centroid_y = (y1 + y2) // 2
        return (centroid_x, centroid_y)

    def show_object(self, frame, object_id, bbox, class_name, confidence, color, draw_centroid=False):
        """Show the object on the frame with the bounding box, ID, confidence, and optional centroid."""
        x1, y1, x2, y2 = map(int, bbox)

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Display the object details (Car or Plate)
        cv2.putText(frame, f"{class_name} ID: {object_id}, Conf: {confidence:.2f}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)

        # Draw the centroid if requested
        if draw_centroid:
            centroid_x, centroid_y = self.get_centroid(bbox)
            cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
            cv2.putText(frame, "Centroid", (centroid_x - 20, centroid_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def _mouse_event_debug(self, event, x, y, flags, frame):
        """Tangani event klik mouse dalam mode debug."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Tambahkan titik yang diklik ke dalam daftar clicked_points
            self.clicked_points.append((x, y))
            print(f"Clicked coordinates: ({x}, {y})")

            # Konversi koordinat piksel menjadi titik normalisasi
            normalized_points = convert_bbox_to_decimal(frame.shape[:2], self.clicked_points)
            print_normalized_points(normalized_points)

            # Gambar titik dan garis berdasarkan klik pengguna
            self.draw_points_and_lines(frame)

            # Tampilkan frame dengan titik dan garis yang diperbarui
            show_cam(f"FLOOR {self.floor_id}: {self.cam_id}", frame)


    # def _mouse_event_debug(self, event, x, y, flags, param):
    #     """Handle mouse events for debugging."""
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         self.clicked_points.append((x, y))
    #         print(f"Clicked coordinates: ({x}, {y})")


    # def _mouse_event_debug(self, event, x, y, flags, frame):
    #     """Handle mouse events in debug mode."""
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         self.clicked_points.append((x, y))
    #         print(f"Clicked coordinates: ({x}, {y})")


    #         normalized_points = convert_bbox_to_decimal((frame.shape[:2]), [self.clicked_points])
    #         print_normalized_points(normalized_points)

    #         self.draw_points_and_lines(frame, self.clicked_points)
    #         show_cam(f"FLOOR {self.floor_id}: {self.cam_id}", frame)

    # def show_display(self, frame, cropped_frame, floor_id, cam_id, tracking_points, poly_bbox, car_info, plate_info, is_centroid_inside, is_debug=True):
    #     self._current_frame = frame.copy()
    #     self.floor_id, self.cam_id = floor_id, cam_id
    #     height, width = frame.shape[:2]

    #     slot = self.db_floor.get_slot_by_id(floor_id)
    #     total_slot, vehicle_total = slot["slot"], slot["vehicle_total"]

    #     draw_tracking_points(frame, tracking_points, (height, width))

    #     last_plate_no = self.db_vehicle_history.get_vehicle_history_by_floor_id(floor_id)["plate_no"]
    #     plate_no = last_plate_no if last_plate_no else ""

    #     # add_overlay(frame, floor_id, cam_id, poly_points, plate_no, total_slot, vehicle_total)

    #     frame = self.show_polygon_area(frame, poly_bbox, is_centroid_inside)

    #     show_yolo(cropped_frame, car_info, color=(255, 255, 255), draw_centroid=True)
    #     if is_centroid_inside:
    #         show_yolo(cropped_frame, plate_info, color=(255, 255, 255))

    #     window_name = f"FLOOR {floor_id}: {cam_id}"
    #     show_cam(window_name, frame)

    #     if is_debug:
    #         self.draw_points_and_lines(frame)
    #         draw_box(frame=frame, boxes=self.car_bboxes)
    #         show_yolo(frame, self.plate_info, color=(0, 0, 255))

    #         cv2.setMouseCallback(
    #             window_name, 
    #             self._mouse_event_debug, 
    #             param=frame
    #         )
            
    #     else:
    #         draw_box(frame=frame, boxes=self.car_bboxes)

    def show_display(
        self, frame, cropped_frame, floor_id, cam_id, 
        tracking_points, poly_bbox, car_info, plate_info, 
        is_centroid_inside, is_debug=True
    ):
        """Tampilkan frame utama dengan berbagai elemen visual."""
        # Salin frame untuk digunakan kembali
        self._current_frame = frame.copy()
        self.floor_id, self.cam_id = floor_id, cam_id
        height, width = frame.shape[:2]

        # Ambil informasi slot dan kendaraan
        slot = self.db_floor.get_slot_by_id(floor_id)
        total_slot, vehicle_total = slot["slot"], slot["vehicle_total"]

        # Gambar titik dan garis tracking
        draw_tracking_points(frame, tracking_points, (height, width))

        # Ambil nomor plat terakhir dari riwayat kendaraan
        last_plate_no = self.db_vehicle_history.get_vehicle_history_by_floor_id(floor_id)["plate_no"]
        plate_no = last_plate_no if last_plate_no else ""

        # Gambar area poligon
        frame = self.show_polygon_area(frame, poly_bbox, is_centroid_inside)

        # Tampilkan bounding box dan centroid kendaraan
        show_yolo(cropped_frame, car_info, color=(255, 255, 255), draw_centroid=True)
        if is_centroid_inside:
            show_yolo(cropped_frame, plate_info, color=(255, 255, 255))

        # Tentukan nama jendela untuk kamera saat ini
        window_name = f"FLOOR {floor_id}: {cam_id}"
        show_cam(window_name, frame)

        # Jika debug mode aktif, tampilkan lebih banyak elemen
        if is_debug:
            # Gambar titik dan garis berdasarkan klik pengguna
            self.draw_points_and_lines(frame)

            # Gambar bounding box mobil
            draw_box(frame=frame, boxes=self.car_bboxes)

            # Tampilkan info plat dengan bounding box merah
            show_yolo(frame, self.plate_info, color=(0, 0, 255))

            # Set mouse callback untuk menangkap klik dan menambahkan titik
            cv2.setMouseCallback(
                window_name,
                lambda event, x, y, flags, param: self._mouse_event_debug(event, x, y, flags, frame)
            )
        else:
            # Tampilkan bounding box tanpa elemen tambahan
            draw_box(frame=frame, boxes=self.car_bboxes)

    def is_valid_cropped_plate(self, cropped_plate):
        """Check if the cropped plate meets the size requirements and save dimensions to a CSV file."""
        height, width = cropped_plate.shape[:2]
        print(f'height: {height} & width: {width}')
        
        # Save height and width to CSV
        # self.save_dimensions_to_csv(height, width)
        self.save_dimensions_to_excel(height, width)

        # if floor_id == 2:
        #     if cam_id == "IN":
        #         if height < 30 or width < 100:
        #             return False
        #         if height >= width:
        #             return False
        #         compare = abs(height - width)
        #         if compare <= 35 or compare >= 80:
        #             return False

        # elif floor_id == 3:
        #     if cam_id == "OUT":
        #         if height < 30 or width < 90 or width >= 150:
        #             return False
        #         if height >= width:
        #             return False
        #         compare = abs(height - width)
        #         if compare <= 35 or compare >= 85: # 120:
        #             return False
        # else:
        #     if height < 30 or width < 90:
        #         return False
        #     if height >= width:
        #         return False
        #     compare = abs(height - width)
        #     if compare <= 35 or compare >= 80:
        #         return False


        if height < 30 or width >= 80 or width <= 70:
            return False
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

    def save_dimensions_to_excel(self, height, width):
        """Save height and width to an Excel file named with the current date."""
        # Get the current date
        current_date = datetime.now()
        filename = current_date.strftime("%Y_%m_%d") + ".xlsx"

        # Create a DataFrame for the new entry
        new_data = pd.DataFrame([[height, width]], columns=['Height', 'Width'])

        # Check if the file already exists
        if os.path.isfile(filename):
            # Append the new data to the existing file
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                new_data.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        else:
            # Create a new file and write the data
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                new_data.to_excel(writer, index=False, sheet_name='Sheet1')

    def save_dimensions_to_csv(self, height, width):
        """Save height and width to a CSV file named with the current date."""
        # Get the current date
        current_date = datetime.now()
        filename = current_date.strftime("%Y_%m_%d") + ".csv"

        # Write height and width to CSV file
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Check if the file is empty to write the header
            file.seek(0, 2)  # Move to the end of the file
            if file.tell() == 0:
                writer.writerow(['Height', 'Width'])  # Write header if file is empty
            writer.writerow([height, width])  # Write height and width

    # def is_valid_cropped_plate(self, cropped_plate):
    #     """Check if the cropped plate meets the size requirements."""
    #     height, width = cropped_plate.shape[:2]
    #     print(f'height: {height} & width: {width}')
    #     if height < 55 or width < 100:
    #         return False
    #     if height >= width:
    #         return False
    #     compare = abs(height - width)
    #     if compare <= 110 or compare >= 400:
    #         return False
    #     return True

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
            # Ensure that box is a list or tuple and contains valid coordinates
            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, x2, y2 = [max(0, min(int(coord), width if i % 2 == 0 else height)) for i, coord in enumerate(box)]
                cropped_plate = frame[y1:y2, x1:x2]

                if cropped_plate.size > 0 and self.is_valid_cropped_plate(cropped_plate):
                # if cropped_plate.size > 0:
                    cropped_plates.append(cropped_plate)

        return cropped_plates

    # def get_cropped_plates(self, frame, boxes):
    #     """
    #     Extract cropped plate images based on bounding boxes.
    #     Args:
    #         frame: The original image/frame.
    #         boxes: List of bounding boxes (each box is [x1, y1, x2, y2]).

    #     Returns:
    #         cropped_plates: List of cropped plate images.
    #     """
    #     height, width, _ = frame.shape
    #     cropped_plates = []

    #     for box in boxes:
    #         x1, y1, x2, y2 = [max(0, min(int(coord), width if i % 2 == 0 else height)) for i, coord in enumerate(box)]
    #         cropped_plate = frame[y1:y2, x1:x2]

    #         if cropped_plate.size > 0 and self.is_valid_cropped_plate(cropped_plate):
    #             cropped_plates.append(cropped_plate)

    #     return cropped_plates


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

    def get_centroid(self, results, line_pos):
        """Calculate centroids from detection results and determine tracking information."""
        # Extract bounding boxes from the results
        boxes = results[0].boxes.xyxy.cpu().tolist()
        detection = []
        
        for result in boxes:
            x1, y1, x2, y2 = map(int, result)
            centroid_x = (x1 + x2) / 2 if line_pos else x1  # Use center X or left X based on line_pos
            centroid_y = y2 if line_pos else (y1 + y2) / 2  # Use bottom Y or center Y based on line_pos
            detection.append([centroid_x, centroid_y])

        # Get tracking estimates if any
        estimate_tracking, self.track_id = self.get_tracking_centroid(detection)
        return detection if estimate_tracking != () else estimate_tracking

    def get_tracking_centroid(self, centroids):
        if not centroids:  # Check if centroids is empty
            return [], []  # Return empty lists for point and ID

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
        self.centroids = self.get_centroid(results, line_pos=False)
        if inverse:
            return end, start
        else:
            return start, end

    def convert_normalized_to_pixel(self, points, img_dims):
        """Convert a list of normalized points to pixel coordinates."""
        height, width = img_dims
        pixel_points = [(int(x * width), int(y * height)) for (x, y) in points]
        return pixel_points
    
    def convert_normalized_to_pixel_line(self, points, img_dims):
        """Convert a list of normalized points to pixel coordinates."""
        if not isinstance(points, list):
            points = [points]  # Pastikan input berupa list

        height, width = img_dims
        pixel_points = [
            (int(x * width), int(y * height)) for (x, y) in points
        ]
        return pixel_points

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

    def check_car_touch_line(self, frame_size, car_info, polygon_area):
        """
        Cek apakah ada centroid objek yang masuk ke dalam area poligon.
        """
        # Konversi koordinat normalisasi ke piksel
        polygon_points = [
            self.convert_normalized_to_pixel_lines(point, frame_size)
            for point in polygon_area
        ]

        # Loop untuk setiap mobil dalam car_info
        for (object_id, confidence, bbox, class_name) in car_info:
            car_id = object_id  # Gunakan object_id sebagai car_id
            centroid = self.get_centroid_object(bbox)

            # Cek apakah centroid berada di dalam poligon
            if self.is_point_in_polygon(centroid, polygon_points):
                # print(f"Object {car_id} entered the area.")
                return True  # Jika ada yang masuk, kembalikan True

        return False  # Tidak ada objek yang masuk

    def is_point_in_polygon(self, point, polygon):
        """
        Algoritma Ray Casting untuk cek apakah titik berada di dalam poligon.
        """
        x, y = point
        n = len(polygon)
        inside = False

        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[(i + 1) % n]

            intersect = ((yi > y) != (yj > y)) and \
                        (x < (xj - xi) * (y - yi) / (yj - yi) + xi)

            if intersect:
                inside = not inside

        return inside

    def convert_normalized_to_pixel_lines(self, point, frame_size):
        """
        Konversi titik normalisasi (0-1) ke koordinat piksel.
        """
        x_norm, y_norm = point
        width, height = frame_size
        return int(x_norm * width), int(y_norm * height)

    def show_polygon_area(self, frame, polygon_points, is_centroid_inside):
        """
        Tampilkan area poligon berbentuk persegi dengan warna sesuai status (hijau/abu-abu).
        """
        overlay = frame.copy()
        color = (0, 255, 0) if is_centroid_inside else (128, 128, 128)
        alpha = 0.5

        polygon_points_pixel = [
            self.convert_normalized_to_pixel_lines(point, self.frame_size)
            for point in polygon_points
        ]
        
        cv2.fillPoly(overlay, [np.array(polygon_points_pixel)], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

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
                        car_boxes.append(bbox)
                        car_info.append((object_id, confidence, bbox, str(class_id)))

        return car_boxes, car_info

    def process_plate(self, results, original_frame, car_boxes, is_save=True):
        """Process the detection of plates within the original frame size and return plate info."""
        plate_info = []
        cropped_plates = []
        height, width = original_frame.shape[:2]
        num_add_size = 5

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

                            plate_y1 = max(plate_y1 - num_add_size, 0)
                            plate_y2 = min(plate_y2 + num_add_size, original_frame.shape[0])
                            plate_x1 = max(plate_x1 - num_add_size, 0)
                            plate_x2 = min(plate_x2 + num_add_size, original_frame.shape[1])

                            cropped_plate = original_frame[plate_y1:plate_y2, plate_x1:plate_x2]
                            if cropped_plate.size > 0 and self.is_valid_cropped_plate(cropped_plate):
                            # if cropped_plate.size > 0 :
                                cropped_plates.append(cropped_plate)

                            # if is_save:
                            #     self.save_cropped_plate([cropped_plate])

                            return plate_info, cropped_plates

        return plate_info, cropped_plates


    def vehicle_detect(self, arduino_idx, frame, floor_id, cam_id, tracking_points, poly_bbox):
        height, width = frame.shape[:2]
        frame_size = (width, height)
        original_frame = frame.copy()

        area_selection = self.convert_normalized_to_pixel(tracking_points, (height, width))
        cropped_frame = self.crop_frame_with_polygon(frame, area_selection)
        cropped_frame_copy = cropped_frame.copy()

        preprocessed_image = self.preprocess(cropped_frame)

        results = self.model.track(preprocessed_image, conf=0.25, persist=True, verbose=False)

        self.car_bboxes, car_info = self.process_car(results)

        for car_bbox in self.car_bboxes:
            self.centroids = self.get_centroid_object(car_bbox)

        direction = self.is_car_out(self.car_bboxes)
        if direction is not None or self.car_direction is None:
            self.car_direction = direction

        for (object_id, confidence, bbox, class_name) in car_info:
            self.object_id = object_id

        is_centroid_inside = self.check_car_touch_line(frame_size, car_info, poly_bbox)

        if is_centroid_inside and not self.is_vehicle_model:
            self.plate_info, plate_frames = self.process_plate(results, cropped_frame_copy, self.car_bboxes)

            for (object_id, confidence, bbox, class_name) in self.plate_info:
                self.plate_bbox = bbox

            if self.object_id != self.prev_object_id:
                self.frame_count_per_object[self.object_id] = 0

            if self.object_id not in self.frame_count_per_object:
                self.frame_count_per_object[self.object_id] = 0

            if self.frame_count_per_object[self.object_id] == self.max_num_frame:
                for plate_frame in plate_frames:
                    self.save_cropped_plate([plate_frame])                

            elif self.frame_count_per_object[self.object_id] < self.max_num_frame:
                for plate_frame in plate_frames:
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

                    return vehicle_plate_data, cropped_frame_copy, is_centroid_inside, car_info

            else:
                print(f"Skipping saving for object_id: {self.object_id}, frame_count: {self.frame_count_per_object[self.object_id]}")

        return {}, cropped_frame_copy, is_centroid_inside, car_info

    def process_video(self):
        """Process the video to detect cars, track directions, and process plates."""
        cap = cv2.VideoCapture(self.video_path)
        
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.resize(frame, (1080, 720))
            height, width = frame.shape[:2]
            self.frame_size = (width, height)

            self.poly_points, self.tracking_points, self.poly_bbox = define_tracking_polygon(
                height=height, width=width, 
                floor_id=self.floor_id, cam_id=self.cam_id
            )

            vehicle_plate_data, cropped_frame, is_centroid_inside, car_info = self.vehicle_detect(arduino_idx=self.arduino_idx, frame=frame, floor_id=self.floor_id, cam_id=self.cam_id, tracking_points=self.tracking_points, poly_bbox=self.poly_bbox)
            # # print("vehicle_plate_data: ", vehicle_plate_data)
            # self.show_display(frame, cropped_frame=cropped_frame, floor_id=self.floor_id, cam_id=self.cam_id, tracking_points=self.tracking_points, poly_bbox=self.poly_bbox, is_centroid_inside=is_centroid_inside, car_info=car_info, plate_info=self.plate_info)

            self.draw_points_and_lines(frame)

            # (Optional) Additional processing logic...
            # Example: Detect vehicles, plates, and display the frame
            self.show_display(frame, cropped_frame=frame.copy(), 
                              floor_id=self.floor_id, cam_id=self.cam_id,
                              tracking_points=self.tracking_points, 
                              poly_bbox=self.poly_bbox, 
                              car_info=[], plate_info=[], 
                              is_centroid_inside=is_centroid_inside, is_debug=True)

            cv2.imshow(f"FLOOR {self.floor_id}: {self.cam_id}", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # model_path = r"C:\Users\DOT\Documents\febri\weights\yolo11n.pt"
    # model_path = r"D:\engine\cv\car-plate-detection\kendaraan.v1i.yolov8\runs\detect\vehicle-plate-model-n\weights\best.pt"
    model_path = r"C:\Users\DOT\Documents\febri\weights\vehicle_plate_model.pt"
    # model_path = r"C:\Users\DOT\Documents\febri\weights\yolov8n.pt"
    # video_path = r'D:\engine\smart_parking\dataset\cctv\z.mp4'
    # video_path = r'D:\engine\cv\dataset_editor\editor\compose_video.mp4'

    FLOOR_ID = 3
    CAM_ID = "OUT"
    IS_VEHICLE_MODEL = False
    IS_CAMERA = False

    if IS_CAMERA:
        CAM_SOURCE_LT = {
            2: {
                "IN": 'rtsp://admin:Passw0rd@192.168.1.10',
                "OUT": 'rtsp://admin:Passw0rd@192.168.1.11'
            },
            3: {
                "IN": 'rtsp://admin:Passw0rd@192.168.1.12',
                "OUT": 'rtsp://admin:Passw0rd@192.168.1.13'
            },
            4: {
                "IN": 'rtsp://admin:Passw0rd@192.168.1.14',
                "OUT": 'rtsp://admin:Passw0rd@192.168.1.15'
            },
            5: {
                "IN": 'rtsp://admin:Passw0rd@192.168.1.16',
                "OUT": 'rtsp://admin:Passw0rd@192.168.1.27'
            }
        }

        try:
            video_path = CAM_SOURCE_LT[FLOOR_ID][CAM_ID]
        except KeyError:
            raise ValueError(f"Invalid FLOOR_ID {FLOOR_ID} or CAM_ID {CAM_ID}")
    
    else:
        if FLOOR_ID == 2:
            if CAM_ID == "IN":
                # video_path = r"C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4"
                # video_path = r"C:\Users\DOT\Web\RecordFiles\2024-10-22\day\192.168.1.10_01_20241022164924927.mp4"
                video_path = r"C:\Users\DOT\Web\RecordFiles\2024-10-24\CAR\F2_IN_192.168.1.10_01_20241024142756346.mp4"

            else:
                # video_path = r'C:\Users\DOT\Documents\febri\github\combined_video_out.mp4'
                # video_path = r"C:\Users\DOT\Web\RecordFiles\2024-10-22\day\192.168.1.11_01_2024102217293382.mp4"
                # video_path = r"C:\Users\DOT\Web\RecordFiles\2024-10-22\192.168.1.11_01_20241022165758745.mp4"
                # video_path = r"C:\Users\DOT\Web\RecordFiles\2024-10-24\CAR\F2_OUT_192.168.1.11_01_20241024142707319.mp4"
                video_path = r"C:\Users\DOT\Web\RecordFiles\2024-10-24\192.168.1.11_01_20241024170751959.mp4"
        elif FLOOR_ID == 3:
            if CAM_ID == "IN":
                # video_path = r'C:\Users\DOT\Web\RecordFiles\2024-10-22\day\192.168.1.12_01_20241022164946751.mp4'
                # video_path = r"C:\Users\DOT\Documents\febri\video\sequence\LT_3_IN.mp4"
                video_path = r"C:\Users\DOT\Web\RecordFiles\2024-10-24\CAR\F3_IN_192.168.1.12_01_20241024142828799.mp4" # F3 IN
            else:
                # video_path = r'C:\Users\DOT\Web\RecordFiles\2024-10-22\day\192.168.1.11_01_20241022171905925.mp4'
                # video_path = r"C:\Users\DOT\Documents\febri\video\sequence\LT_3_OUT.mp4"
                # video_path = r"C:\Users\DOT\Web\RecordFiles\2024-10-24\CAR\F3_OUT_192.168.1.13_01_20241024142951212.mp4"
                # video_path = r"C:\Users\DOT\Web\RecordFiles\2024-10-25\192.168.1.13_01_20241025152631450.mp4"
                video_path = r"C:\Users\DOT\Documents\febri\video\F3_OUT_20241025_1.mp4"
        elif FLOOR_ID == 4:
            if CAM_ID == "IN":
                video_path = r"C:\Users\DOT\Documents\febri\video\sequence\LT_4_IN.mp4"
            else:
                video_path = r"C:\Users\DOT\Documents\febri\video\sequence\LT_4_OUT.mp4"
        elif FLOOR_ID == 5:
            if CAM_ID == "IN":
                video_path = r"C:\Users\DOT\Documents\febri\video\sequence\LT_5_IN.mp4"
            else:
                video_path = r"C:\Users\DOT\Documents\febri\video\sequence\LT_5_OUT.mp4"


    detector = VehicleDetector(model_path, video_path, floor_id=FLOOR_ID, cam_id=CAM_ID, is_vehicle_model=IS_VEHICLE_MODEL)
    detector.process_video()