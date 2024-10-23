import cv2
import numpy as np

from src.controllers.utils.util import (
    convert_normalized_to_pixel_lines,
    convert_normalized_to_pixel,
    get_centroid
)

from src.Integration.service_v1.controller.plat_controller import PlatController
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.fetch_api_controller import FetchAPIController
from src.Integration.service_v1.controller.vehicle_history_controller import VehicleHistoryController

db_plate = PlatController()
db_floor = FloorController()
db_mysn = FetchAPIController()
db_vehicle_history = VehicleHistoryController()



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

def print_normalized_points(normalized_points):
    print("[")
    for point in normalized_points:
        if isinstance(point[0], tuple):  # If `point` is a list of tuples (a polygon)
            for sub_point in point:
                print(f"    ({sub_point[0]:.16f}, {sub_point[1]:.16f}),")
        else:  # If `point` is a flat tuple (x, y)
            print(f"    ({point[0]:.16f}, {point[1]:.16f}),")
    print("]")

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

def add_overlay_v6(frame, floor_id, cam_id, poly_points, plate_no, total_slot, vehicle_total):
    """Add text and lines to the frame."""
    show_text(f"Floor : {floor_id} {cam_id}", frame, 5, 50)
    show_text(f"Plate No. : {plate_no}", frame, 5, 100)
    color = (0, 255, 0) if total_slot > 0 else (0, 0, 255)
    show_text(f"P. Spaces Available : {total_slot}", frame, 5, 150, color)
    show_text(f"Car Total : {vehicle_total}", frame, 5, 200)
    # show_line(frame, poly_points[0], poly_points[1])
    # show_line(frame, poly_points[2], poly_points[3])

def add_overlay(frame, floor_id, cam_id, poly_points, plate_no, total_slot, vehicle_total, poly_bbox):
    """Add text and lines to the frame."""
    show_text(f"Floor : {floor_id} {cam_id}", frame, 5, 50)
    show_text(f"Plate No. : {plate_no}", frame, 5, 100)
    color = (0, 255, 0) if total_slot > 0 else (0, 0, 255)
    show_text(f"P. Spaces Available : {total_slot}", frame, 5, 150, color)
    show_text(f"Car Total : {vehicle_total}", frame, 5, 200)
    # show_line(frame, poly_points[0], poly_points[1])
    # show_line(frame, poly_points[2], poly_points[3])

    show_polygon_area(frame=frame, poly_bbox=poly_bbox, is_centroid_inside=False)

def draw_points_and_lines(frame, clicked_points):
    for point in clicked_points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
    if len(clicked_points) > 1:
        for i in range(len(clicked_points)):
            start_point = clicked_points[i]
            end_point = clicked_points[(i + 1) % len(clicked_points)]
            cv2.line(frame, start_point, end_point, (255, 0, 0), 1)

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


def draw_box(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        color = (255, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
    
    return frame

def draw_tracking_points(frame, points, img_dims):
    pixel_points = convert_normalized_to_pixel(points, img_dims)

    for i in range(len(pixel_points)):
        start_point = pixel_points[i]
        end_point = pixel_points[(i + 1) % len(pixel_points)]
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    for point in pixel_points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

def show_object(frame, object_id, bbox, class_name, confidence, color, draw_centroid=False):
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
        centroid_x, centroid_y = get_centroid(bbox)
        cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
        cv2.putText(frame, "Centroid", (centroid_x - 20, centroid_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def show_polygon_area(frame, poly_bbox, is_centroid_inside):
    """
    Tampilkan area poligon berbentuk persegi dengan warna sesuai status (hijau/abu-abu).
    """
    height, width = frame.shape[:2]
    frame_size = (width, height)    
    overlay = frame.copy()

    # Pilih warna sesuai status
    color = (0, 255, 0) if is_centroid_inside else (128, 128, 128)  # Hijau atau Abu-abu
    alpha = 0.5  # Transparansi

    # Konversi titik polygon ke koordinat piksel
    polygon_points_pixel = [
        convert_normalized_to_pixel_lines(point, frame_size)
        for point in poly_bbox
    ]
    
    # Gambar bidang poligon berdasarkan polygon_points
    cv2.fillPoly(overlay, [np.array(polygon_points_pixel)], color)

    # Gabungkan overlay dengan frame asli menggunakan transparansi
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame

def show_display(frame, cropped_frame, floor_id, cam_id, tracking_points, plate_no, poly_bbox, car_info, plate_info, is_centroid_inside, is_debug=True):
    car_bbox = []
    clicked_points = []

    floor_id, cam_id = floor_id, cam_id
    height, width = frame.shape[:2]

    slot = db_floor.get_slot_by_id(floor_id)
    total_slot, vehicle_total = slot["slot"], slot["vehicle_total"]

    draw_tracking_points(frame, tracking_points, (height, width))

    last_plate_no = db_vehicle_history.get_vehicle_history_by_floor_id(floor_id)["plate_no"]
    plate_no = last_plate_no if last_plate_no else ""

    # add_overlay(frame, floor_id, cam_id, poly_points, plate_no, total_slot, vehicle_total)

    frame = show_polygon_area(frame, poly_bbox, is_centroid_inside)

    show_yolo(cropped_frame, car_info, color=(255, 255, 255), draw_centroid=True)
    if is_centroid_inside:
        show_yolo(cropped_frame, plate_info, color=(255, 255, 255))

    for (object_id, confidence, bbox, class_name) in car_info:
        car_bbox = bbox

    window_name = f"FLOOR {floor_id}: {cam_id}"
    show_cam(window_name, frame)

    # if is_debug:
    #     draw_points_and_lines(frame, clicked_points)
    #     draw_box(frame=frame, boxes=car_bbox)
    #     show_yolo(frame, plate_info, color=(0, 0, 255))

    #     cv2.setMouseCallback(
    #         window_name, 
    #         self._mouse_event_debug, 
    #         param=frame
    #     )
        
    # else:
    #     draw_box(frame=frame, boxes=car_bbox)