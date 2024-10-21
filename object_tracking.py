import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from shapely.geometry import Polygon

from src.config.config import config
from src.Integration.service_v1.controller.plat_controller import PlatController
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.fetch_api_controller import FetchAPIController
from src.Integration.service_v1.controller.vehicle_history_controller import VehicleHistoryController
from utils.centroid_tracking import CentroidTracker

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
    # normalized_polygons = [
    #     # ((bbox[0] / width, bbox[1] / height), (bbox[2] / width, bbox[3] / height)) for bbox in polygons
    #     (bbox[0] / width, bbox[1] / height) for bbox in polygons
    # ]

    normalized_polygons = []
    for bbox in polygons:
        if isinstance(bbox, list):  # If it's a list of tuples
            normalized_bbox = [(pt[0] / width, pt[1] / height) for pt in bbox]
        else:  # If it's a simple tuple (x, y, w, h)
            normalized_bbox = (bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height)
        normalized_polygons.append(normalized_bbox)

    return normalized_polygons

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

def add_overlay(frame, floor_id, cam_id, poly_points, plate_no, total_slot, vehicle_total):
    show_text(f"Floor : {floor_id} {cam_id}", frame, 5, 50)
    show_text(f"Plate No. : {plate_no}", frame, 5, 100)
    color = (0, 255, 0) if total_slot > 0 else (0, 0, 255)
    show_text(f"P. Spaces Available : {total_slot}", frame, 5, 150, color)
    show_text(f"Car Total : {vehicle_total}", frame, 5, 200)
    show_line(frame, poly_points[0], poly_points[1])
    show_line(frame, poly_points[2], poly_points[3])

def draw_points_and_lines(frame, clicked_points):
    for point in clicked_points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
    if len(clicked_points) > 1:
        for i in range(len(clicked_points)):
            start_point = clicked_points[i]
            end_point = clicked_points[(i + 1) % len(clicked_points)]
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

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

def define_tracking_polygon(frame, height, width, floor_id, cam_id):
    if frame.shape[0] == () or frame.shape[1] == ():
        return "", np.array([]), np.array([])

    if floor_id == 2:
        tracking_point = config.TRACKING_POINT2_F1_IN if cam_id == "IN" else config.TRACKING_POINT2_F1_OUT
        polygon_point = config.POLYGON_POINT_LT2_IN if cam_id == "IN" else config.POLYGON_POINT_LT2_OUT
    elif floor_id == 3:
        tracking_point = config.TRACKING_POINT2_F1_IN if cam_id == "IN" else config.TRACKING_POINT2_F1_OUT
        polygon_point = config.POLYGON_POINT_LT3_IN if cam_id == "IN" else config.POLYGON_POINT_LT3_OUT
    elif floor_id == 4:
        tracking_point = config.TRACKING_POINT2_F1_IN if cam_id == "IN" else config.TRACKING_POINT2_F1_OUT
        polygon_point = config.POLYGON_POINT_LT4_IN if cam_id == "IN" else config.POLYGON_POINT_LT4_OUT
    elif floor_id == 5:
        tracking_point = config.TRACKING_POINT2_F1_IN if cam_id == "IN" else config.TRACKING_POINT2_F1_OUT
        polygon_point = config.POLYGON_POINT_LT5_IN if cam_id == "IN" else config.POLYGON_POINT_LT5_OUT
    else:
        return "", np.array([]), np.array([])

    poly_points = convert_decimal_to_bbox((height, width), polygon_point)
    
    return poly_points, tracking_point, frame

def point_position(line, line2, point, inverse=False):
    x1, y1 = line
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
    def __init__(self, model_path, video_path, output_folder="output"):
        self.centroid_tracking = CentroidTracker(maxDisappeared=75)
        self.class_names = ['car', 'plate', 'truck']
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_folder = output_folder

        self.car_direction = None
        self.prev_centroid = None
        self.num_skip_centroid = 0

        self.tracking_points = config.TRACKING_POINT2_F1_IN
        self.poly_points = config.POLYGON_POINT_LT2_IN

        self.clicked_points = []
        self.car_bboxes = []

        self.db_plate = PlatController()
        self.db_floor = FloorController()
        self.db_mysn = FetchAPIController()
        self.db_vehicle_history = VehicleHistoryController()

        os.makedirs(self.output_folder, exist_ok=True)

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

    # def get_centroid(bbox):
    #     """Calculate the centroid of a bounding box."""
    #     x1, y1, x2, y2 = map(int, bbox)
    #     centroid_x = (x1 + x2) // 2
    #     centroid_y = (y1 + y2) // 2
    #     return (centroid_x, centroid_y)

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

    def process_frame(self, frame, floor_id, cam_id, is_debug=True):
        self._current_frame = frame.copy()
        self.floor_id, self.cam_id = floor_id, cam_id
        height, width = frame.shape[:2]

        slot = self.db_floor.get_slot_by_id(floor_id)
        total_slot, vehicle_total = slot["slot"], slot["vehicle_total"]

        self.poly_points, self.tracking_points, frame = define_tracking_polygon(
            frame=frame, height=height, width=width, 
            floor_id=floor_id, cam_id=cam_id
        )

        draw_tracking_points(frame, self.tracking_points, (height, width))

        last_plate_no = self.db_vehicle_history.get_vehicle_history_by_floor_id(floor_id)["plate_no"]
        plate_no = last_plate_no if last_plate_no else ""

        # add_overlay(frame, floor_id, cam_id, self.poly_points, plate_no, total_slot, vehicle_total)

        if is_debug:
            draw_points_and_lines(frame, self.clicked_points)
            draw_box(frame=frame, boxes=self.car_bboxes)
        else:
            draw_box(frame=frame, boxes=self.car_bboxes)

        window_name = f"FLOOR {floor_id}: {cam_id}"
        show_cam(window_name, frame)

        cv2.setMouseCallback(
            window_name, 
            self._mouse_event_debug if is_debug else self._mouse_event, 
            param=frame
        )

    def show_yolo(self, frame, object_info, color, draw_centroid=False):
        """Display YOLO-detected objects (either cars or plates) on the frame."""
        for (object_id, confidence, bbox, class_name) in object_info:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, bbox)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display the object details (ID, confidence)
            cv2.putText(frame, f"{class_name} ID: {object_id}, Conf: {confidence:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2)

            # Optionally draw the centroid
            if draw_centroid:
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
                cv2.putText(frame, "Centroid", (centroid_x - 20, centroid_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def process_car(self, results, frame):
        """Process the detection of cars and return car information."""
        car_boxes = []
        car_info = []  # Store car object info for return
        for r in results[0].boxes:
            if r.id is not None and r.cls is not None and r.conf is not None:
                class_id = int(r.cls.item())
                object_id = int(r.id.item())
                bbox = r.xyxy[0].cpu().numpy().tolist()
                confidence = float(r.conf.item())
                class_name = self.class_names[class_id]

                if class_name == "car":
                    car_boxes.append(bbox)
                    car_info.append((object_id, confidence, bbox, class_name))  # Collect car data

        return car_boxes, car_info

    def process_plate(self, results, frame, original_frame, car_boxes, is_save=True):
        """Process the detection of plates within car bounding boxes and return plate info."""
        plate_info = []  # Store plate object info for return
        for r in results[0].boxes:
            if r.id is not None and r.cls is not None and r.conf is not None:
                class_id = int(r.cls.item())
                class_name = self.class_names[class_id]

                if class_name == "plate":
                    bbox = r.xyxy[0].cpu().numpy().tolist()
                    confidence = float(r.conf.item())
                    plate_x1, plate_y1, plate_x2, plate_y2 = map(int, bbox)

                    # Check if plate is inside any car bounding box
                    for car_box in car_boxes:
                        car_x1, car_y1, car_x2, car_y2 = map(int, car_box)

                        if (car_x1 <= plate_x1 <= car_x2 and car_y1 <= plate_y1 <= car_y2) and \
                           (car_x1 <= plate_x2 <= car_x2 and car_y1 <= plate_y2 <= car_y2):
                            plate_info.append((r.id.item(), confidence, bbox, class_name))  # Collect plate data
                            # self.save_plate(original_frame, bbox, is_save)

        return plate_info

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


    def process_video(self, is_save=True):
        """Process the video to detect cars, track directions, and process plates."""
        cap = cv2.VideoCapture(self.video_path)
        
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.resize(frame, (1080, 720))
            height, width = frame.shape[:2]
            original_frame = frame.copy()

            self.process_frame(frame, floor_id=2, cam_id="IN", is_debug=True)

            area_selection = self.convert_normalized_to_pixel(self.tracking_points, (height, width))

            cropped_frame = self.crop_frame_with_polygon(frame, area_selection)
            if cropped_frame is None or cropped_frame.size == 0:
                print("Cropped frame is empty.")
                continue

            results = self.model.track(cropped_frame, persist=True, verbose=False)
            self.car_bboxes, car_info = self.process_car(results, cropped_frame)
            # self.show_yolo(cropped_frame, car_info, color=(0, 255, 0), draw_centroid=True)
            self.centroids = self.get_centroid(results, line_pos=True)

            direction = self.is_car_out(self.car_bboxes)
            if direction is not None or self.car_direction is None:
                self.car_direction = direction
                print("Car direction: ", self.car_direction)

            # # Convert polygon points to pixel coordinates
            # pixel_polygon_points = convert_normalized_to_pixel(self.poly_points, (height, width))
            # show_line(cropped_frame, pixel_polygon_points[0], pixel_polygon_points[1])
            # show_line(cropped_frame, pixel_polygon_points[2], pixel_polygon_points[3])

            start_line, end_line = self.check_centroid_location(
                results, self.poly_points, inverse=self.car_direction
            )

            if start_line and end_line:
                print(f'Start line: {start_line}, End line: {end_line}')

                plate_info = self.process_plate(results, frame, original_frame, self.car_bboxes, is_save)

                # self.show_yolo(frame, plate_info, color=(0, 0, 255))

            # cv2.imshow("Cropped YOLOv8 Tracking", cropped_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    # model_path = r"C:\Users\DOT\Documents\febri\weights\yolo11n.pt"
    model_path = r"D:\engine\cv\car-plate-detection\kendaraan.v1i.yolov8\runs\detect\vehicle-plate-model-n\weights\best.pt"
    video_path = r'D:\engine\smart_parking\dataset\cctv\z.mp4'
    # video_path = r'D:\engine\cv\dataset_editor\editor\compose_video.mp4'

    detector = VehicleDetector(model_path, video_path)
    detector.process_video(is_save=True)