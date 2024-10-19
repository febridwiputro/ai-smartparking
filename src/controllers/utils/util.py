import cv2
import numpy as np
import Levenshtein as lev
import logging

from src.config.config import config
from src.config.logger import logger
from src.Integration.service_v1.controller.plat_controller import PlatController
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.fetch_api_controller import FetchAPIController
from src.Integration.service_v1.controller.vehicle_history_controller import VehicleHistoryController
from src.view.show_cam import show_cam, show_text, show_line
from src.controllers.matrix_controller import MatrixController

db_plate = PlatController()
db_floor = FloorController()
db_mysn = FetchAPIController()
db_vehicle_history = VehicleHistoryController()

import cv2
import numpy as np
import logging
from colorama import Fore, Style, init

init(autoreset=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_background(gray_image, verbose=False):
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    white_pixels = np.sum(binary_image == 255)
    black_pixels = np.sum(binary_image == 0)

    bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

    green_pixels = np.sum(mask_green == 255)
    yellow_pixels = np.sum(mask_yellow == 255)
    red_pixels = np.sum(mask_red == 255)
    blue_pixels = np.sum(mask_blue == 255)

    if abs(black_pixels - blue_pixels) / max(black_pixels, blue_pixels) < 0.1:
        if verbose:
            logging.info('=' * 80)
            logging.info('=' * 30 + " BLACK-BLUE PLATE " + '=' * 30)
            logging.info('=' * 80)
        return "bg_black_blue"

    if white_pixels > black_pixels and white_pixels > green_pixels and white_pixels > yellow_pixels and white_pixels > red_pixels and white_pixels > blue_pixels:
        if verbose:
            logging.info('=' * 80)
            logging.info('=' * 30 + " WHITE PLATE " + '=' * 30)
            logging.info('=' * 80)
        return "bg_white"

    elif black_pixels > white_pixels and black_pixels > green_pixels and black_pixels > yellow_pixels and black_pixels > red_pixels and black_pixels > blue_pixels:
        if verbose:
            logging.info('=' * 80)
            logging.info('=' * 30 + " BLACK PLATE " + '=' * 30)
            logging.info('=' * 80)
        return "bg_black"

    elif green_pixels > white_pixels and green_pixels > yellow_pixels and green_pixels > red_pixels and green_pixels > blue_pixels:
        return "bg_green"

    elif yellow_pixels > white_pixels and yellow_pixels > green_pixels and yellow_pixels > red_pixels and yellow_pixels > blue_pixels:
        return "bg_yellow"

    elif red_pixels > white_pixels and red_pixels > green_pixels and red_pixels > yellow_pixels and red_pixels > blue_pixels:
        return "bg_red"

    elif blue_pixels > white_pixels and blue_pixels > green_pixels and blue_pixels > yellow_pixels and blue_pixels > red_pixels:
        return "bg_blue"

    else:
        return "bg_unknown"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_background(gray_image, verbose=False):
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    white_pixels = np.sum(binary_image == 255)
    black_pixels = np.sum(binary_image == 0)

    bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

    green_pixels = np.sum(mask_green == 255)
    yellow_pixels = np.sum(mask_yellow == 255)
    red_pixels = np.sum(mask_red == 255)
    blue_pixels = np.sum(mask_blue == 255)

    if abs(black_pixels - blue_pixels) / max(black_pixels, blue_pixels) < 0.1:
        if verbose:
            logging.info('=' * 80)
            logging.info('=' * 30 + " BLACK-BLUE PLATE " + '=' * 30)
            logging.info('=' * 80)
        return "bg_black_blue"

    if white_pixels > black_pixels and white_pixels > green_pixels and white_pixels > yellow_pixels and white_pixels > red_pixels and white_pixels > blue_pixels:
        if verbose:
            logging.info('=' * 80)
            logging.info('=' * 30 + " WHITE PLATE " + '=' * 30)
            logging.info('=' * 80)
        return "bg_white"

    elif black_pixels > white_pixels and black_pixels > green_pixels and black_pixels > yellow_pixels and black_pixels > red_pixels and black_pixels > blue_pixels:
        if verbose:
            logging.info('=' * 80)
            logging.info('=' * 30 + " BLACK PLATE " + '=' * 30)
            logging.info('=' * 80)
        return "bg_black"

    elif green_pixels > white_pixels and green_pixels > yellow_pixels and green_pixels > red_pixels and green_pixels > blue_pixels:
        return "bg_green"

    elif yellow_pixels > white_pixels and yellow_pixels > green_pixels and yellow_pixels > red_pixels and yellow_pixels > blue_pixels:
        return "bg_yellow"

    elif red_pixels > white_pixels and red_pixels > green_pixels and red_pixels > yellow_pixels and red_pixels > blue_pixels:
        return "bg_red"

    elif blue_pixels > white_pixels and blue_pixels > green_pixels and blue_pixels > yellow_pixels and blue_pixels > red_pixels:
        return "bg_blue"

    else:
        return "bg_unknown"

def check_floor(cam_idx):
    cam_map = {
        0: (2, "IN"), 1: (2, "OUT"),
        2: (3, "IN"), 3: (3, "OUT"),
        4: (4, "IN"), 5: (4, "OUT"),
        6: (5, "IN"), 7: (5, "OUT")
    }
    return cam_map.get(cam_idx, (0, ""))

def find_closest_strings_dict(target, strings):
    distances = np.array([lev.distance(target, s) for s in strings])
    min_distance = np.min(distances)
    min_indices = np.where(distances == min_distance)[0]
    closest_strings_dict = {strings[i]: distances[i] for i in min_indices}
    return closest_strings_dict

def most_freq(lst):
    return max(set(lst), key=lst.count) if lst else ""

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

def crop_polygon(image, points):
    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1, 1, 2))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], (255,255,255))
    result = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(points)
    cropped_image = result[y:y + h, x:x + w]
    return cropped_image

def convert_normalized_to_pixel(points, img_dims):
    """Convert a list of normalized points to pixel coordinates."""
    height, width = img_dims
    pixel_points = [(int(x * width), int(y * height)) for (x, y) in points]
    return pixel_points

def draw_tracking_points(frame, points, img_dims):
    """Draw lines connecting a list of normalized points to form a polygon."""

    pixel_points = convert_normalized_to_pixel(points, img_dims)

    for i in range(len(pixel_points)):
        start_point = pixel_points[i]
        end_point = pixel_points[(i + 1) % len(pixel_points)]
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    for point in pixel_points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

def add_overlay(frame, floor_id, cam_id, poly_points, plate_no, total_slot, vehicle_total):
    """Add text and lines to the frame."""
    show_text(f"Floor : {floor_id} {cam_id}", frame, 5, 50)
    show_text(f"Plate No. : {plate_no}", frame, 5, 100)
    color = (0, 255, 0) if total_slot > 0 else (0, 0, 255)
    show_text(f"P. Spaces Available : {total_slot}", frame, 5, 150, color)
    show_text(f"Car Total : {vehicle_total}", frame, 5, 200)
    show_line(frame, poly_points[0], poly_points[1])
    show_line(frame, poly_points[2], poly_points[3])

def draw_points_and_lines(frame, clicked_points):
    """Draw all clicked points and connect them to form a closed shape."""
    for point in clicked_points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

    if len(clicked_points) > 1:
        for i in range(len(clicked_points)):
            start_point = clicked_points[i]
            end_point = clicked_points[(i + 1) % len(clicked_points)]
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)


def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scale = max_width / width if width > height else max_height / height
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return image


def crop_frame(frame, height, width, floor_id, cam_id):
    # polygons_point = [ config.POINTS_BACKGROUND_LT2_OUT]

    # polygons_point = [config.POINTS_BACKGROUND_LT2_IN, 
    #                   config.POINTS_BACKGROUND_LT2_OUT,
    #                   config.POINTS_BACKGROUND_LT3_IN,
    #                   config.POINTS_BACKGROUND_LT3_OUT,
    #                   config.POINTS_BACKGROUND_LT4_IN,
    #                   config.POINTS_BACKGROUND_LT4_OUT,
    #                   config.POINTS_BACKGROUND_LT5_IN,
    #                   config.POINTS_BACKGROUND_LT5_OUT]

    # polygons_point = [config.TRACKING_POINT2_F1_IN]
    
    # point=polygons_point[cam_id]

    # polygons = [point]
    # bbox = convert_decimal_to_bbox((height, width), polygons)
    # frame = crop_polygon(frame, bbox[0])

    if frame.shape[0] == () or frame.shape[1] == ():
        return "", np.array([]), np.array([])

    if floor_id == 2:
        # polygon_point = config.POLYGON_POINT_LT2_OUT
        tracking_point = config.TRACKING_POINT2_F1_IN if cam_id == "IN" else config.TRACKING_POINT2_F1_OUT
        polygon_point = config.POLYGON_POINT_LT2_IN if cam_id == "IN" else config.POLYGON_POINT_LT2_OUT
        
        # tracking_point = config.TRACKING_POINT2_F1_IN if cam_id == "IN" else config.TRACKING_POINT2_F1_IN
        # polygon_point = config.POLYGON_POINT_LT2_IN if cam_id == "IN" else config.POLYGON_POINT_LT2_IN
    elif floor_id == 3:
        tracking_point = config.TRACKING_POINT2_F1_IN if cam_id == "IN" else config.TRACKING_POINT2_F1_OUT
        polygon_point = config.POLYGON_POINT_LT3_IN if cam_id == "IN" else config.POLYGON_POINT_LT3_OUT
        # tracking_point = config.TRACKING_POINT2_F1_IN if cam_id == "IN" else config.TRACKING_POINT2_F1_IN
        # polygon_point = config.POLYGON_POINT_LT2_IN if cam_id == "IN" else config.POLYGON_POINT_LT2_IN
    elif floor_id == 4:
        tracking_point = config.TRACKING_POINT2_F1_IN if cam_id == "IN" else config.TRACKING_POINT2_F1_OUT
        polygon_point = config.POLYGON_POINT_LT4_IN if cam_id == "IN" else config.POLYGON_POINT_LT4_OUT
        # tracking_point = config.TRACKING_POINT2_F1_IN if cam_id == "IN" else config.TRACKING_POINT2_F1_IN
        # polygon_point = config.POLYGON_POINT_LT2_IN if cam_id == "IN" else config.POLYGON_POINT_LT2_IN
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

# def convert_bbox_to_decimal(img_dims, polygons):
#     height, width = img_dims
#     normalized_polygons = []
#     for pol in polygons:
#         normalized_pol = []
#         for bbox in pol:
#             normalized_bbox = (bbox[0] / width, bbox[1] / height)
#             normalized_pol.append(normalized_bbox)
#         normalized_polygons.append(normalized_pol)
#     return normalized_polygons

def convert_bbox_to_decimal(img_dims, polygons):
    """Convert points to normalized coordinates based on image dimensions."""
    height, width = img_dims
    normalized_polygons = []

    for pol in polygons:
        normalized_pol = [
            (bbox[0] / width, bbox[1] / height) for bbox in pol
        ]
        normalized_polygons.extend(normalized_pol)

    return normalized_polygons

def print_normalized_points(normalized_points):
    """Print normalized points in the desired format."""
    print("[")
    for point in normalized_points:
        print(f"    ({point[0]:.16f}, {point[1]:.16f}),")
    print("]")

# def mouse_event(event, x, y, height, width):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(convert_bbox_to_decimal((height, width), [[[x, y]]]))

def check_db(text):
    if not db_plate.check_exist_plat(license_no=text):
        closest_text = find_closest_strings_dict(text, db_plate.get_all_plat())
        if len(closest_text) == 1 and list(closest_text.values())[0] <= 2:
            text = list(closest_text.keys())[0]
            return True
        else:
            return False
    else:
        # print("plat ada di DB : ", self.text)
        return True

def parking_space_vehicle_counter(floor_id, cam_id, arduino_idx, car_direction, plate_no, container_plate_no, plate_no_is_registered):
    current_floor_position, current_cam_position = floor_id, cam_id
    current_data = db_floor.get_slot_by_id(current_floor_position)
    current_slot = current_data["slot"]
    current_max_slot = current_data["max_slot"]
    current_vehicle_total = current_data["vehicle_total"]
    current_slot_update = current_slot
    current_vehicle_total_update = current_vehicle_total

    prev_floor_position = current_floor_position - 1
    prev_data = db_floor.get_slot_by_id(prev_floor_position)
    prev_slot = prev_data["slot"]
    prev_max_slot = prev_data["max_slot"]
    prev_vehicle_total = prev_data["vehicle_total"]
    prev_slot_update = prev_slot
    prev_vehicle_total_update = prev_vehicle_total

    next_floor_position = current_floor_position - 1
    next_data = db_floor.get_slot_by_id(next_floor_position)
    next_slot = next_data["slot"]
    next_max_slot = next_data["max_slot"]
    next_vehicle_total = next_data["vehicle_total"]
    next_slot_update = next_slot
    next_vehicle_total_update = next_vehicle_total

    get_plate_history = db_vehicle_history.get_vehicle_history_by_plate_no(plate_no=plate_no)
    # print("get_plate_history: ", get_plate_history)

    # NAIK / MASUK
    if not car_direction:
        # if get_plate_history:
        #     if get_plate_history[0]['floor_id'] != current_floor_position:
        #         print(f"Update vehicle history karena floor_id tidak sesuai: {get_plate_history[0]['floor_id']} != {current_floor_position}")
                
        #         # Update vehicle history
        #         update_plate_history = self.db_vehicle_history.update_vehicle_history_by_plate_no(
        #             plate_no=plate_no, 
        #             floor_id=current_floor_position, 
        #             camera=current_cam_position
        #         )

        #         if update_plate_history:
        #             print(f"Vehicle history updated for plate_no: {plate_no} to floor_id: {current_floor_position}")
        #         else:
        #             print(f"Failed to update vehicle history for plate_no: {plate_no}")

        # if get_plate_history:
        #     if get_plate_history[0]['floor_id'] != current_floor_position:
        #         print(f"Update vehicle history karena floor_id tidak sesuai: {get_plate_history[0]['floor_id']} != {current_floor_position}")
                
        #         # Update vehicle history
        #         update_plate_history = self.db_vehicle_history.update_vehicle_history_by_plate_no(
        #             plate_no=plate_no, 
        #             floor_id=current_floor_position, 
        #             camera=current_cam_position
        #         )

        #         if update_plate_history:
        #             print(f"Vehicle history updated for plate_no: {plate_no} to floor_id: {current_floor_position}")
        #         else:
        #             print(f"Failed to update vehicle history for plate_no: {plate_no}")

        #     if current_floor_position == 5 and get_plate_history[0]['floor_id'] == 4:
        #         current_slot_update = current_slot + 1
        #         self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

        #         current_vehicle_total_update = current_vehicle_total - 1
        #         self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)
        #         print(f"Updated current_slot to {current_slot_update} and vehicle_total to {current_vehicle_total_update}")

        #     elif current_floor_position == 4 and get_plate_history[0]['floor_id'] == 3:
        #         current_slot_update = current_slot + 1
        #         self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

        #         current_vehicle_total_update = current_vehicle_total - 1
        #         self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)
        #         print(f"Updated current_slot to {current_slot_update} and vehicle_total to {current_vehicle_total_update}")

        #     elif current_floor_position == 3 and get_plate_history[0]['floor_id'] == 2:
        #         current_slot_update = current_slot + 1
        #         self.db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

        #         current_vehicle_total_update = current_vehicle_total - 1
        #         self.db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)
        #         print(f"Updated current_slot to {current_slot_update} and vehicle_total to {current_vehicle_total_update}")

        # print("VEHICLE - IN")
        # print(f'CURRENT FLOOR : {current_floor_position} && PREV FLOOR {prev_floor_position}')  

        if current_slot == 0:
            # print("UPDATE 0")
            current_slot_update = current_slot
            db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

            current_vehicle_total_update = current_vehicle_total + 1
            db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)

            if prev_floor_position > 1:
                if prev_slot == 0:
                    if prev_vehicle_total > prev_max_slot:
                        prev_slot_update = prev_slot
                        db_floor.update_slot_by_id(id=prev_floor_position, new_slot=prev_slot_update)

                        prev_vehicle_total_update = prev_vehicle_total - 1
                        db_floor.update_vehicle_total_by_id(id=prev_floor_position, new_vehicle_total=prev_vehicle_total_update)
                    else:
                        prev_slot_update = prev_slot + 1
                        db_floor.update_slot_by_id(id=prev_floor_position, new_slot=prev_slot_update)

                        prev_vehicle_total_update = prev_vehicle_total - 1
                        db_floor.update_vehicle_total_by_id(id=prev_floor_position, new_vehicle_total=prev_vehicle_total_update)                            

                elif prev_slot > 0 and prev_slot < prev_max_slot:
                    prev_slot_update = prev_slot + 1
                    db_floor.update_slot_by_id(id=prev_floor_position, new_slot=prev_slot_update)

                    prev_vehicle_total_update = prev_vehicle_total - 1
                    db_floor.update_vehicle_total_by_id(id=prev_floor_position, new_vehicle_total=prev_vehicle_total_update)

        elif current_slot > 0 and current_slot <= current_max_slot:
            current_slot_update = current_slot - 1
            # print("current_slot_update: ", current_slot_update)
            db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

            current_vehicle_total_update = current_vehicle_total + 1
            # print("current_vehicle_total_update: ", current_vehicle_total_update)
            db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)

            if prev_floor_position > 1:
                if prev_slot == 0:
                    # print("IN 1")
                    if prev_vehicle_total > prev_max_slot:
                        prev_slot_update = prev_slot
                        db_floor.update_slot_by_id(id=prev_floor_position, new_slot=prev_slot_update)

                        prev_vehicle_total_update = prev_vehicle_total - 1
                        db_floor.update_vehicle_total_by_id(id=prev_floor_position, new_vehicle_total=prev_vehicle_total_update)
                    else:
                        prev_slot_update = prev_slot + 1
                        db_floor.update_slot_by_id(id=prev_floor_position, new_slot=prev_slot_update)

                        prev_vehicle_total_update = prev_vehicle_total - 1
                        db_floor.update_vehicle_total_by_id(id=prev_floor_position, new_vehicle_total=prev_vehicle_total_update)                            

                elif prev_slot > 0 and prev_slot < prev_max_slot:
                    # print("IN 2")
                    prev_slot_update = prev_slot + 1
                    # print("prev_slot_update: ", prev_slot_update)
                    # print("prev_slot_update: ", prev_slot_update)

                    db_floor.update_slot_by_id(id=prev_floor_position, new_slot=prev_slot_update)

                    prev_vehicle_total_update = prev_vehicle_total - 1
                    # print("prev_vehicle_total_update: ", prev_vehicle_total_update)
                    db_floor.update_vehicle_total_by_id(id=prev_floor_position, new_vehicle_total=prev_vehicle_total_update)

    # TURUN / KELUAR
    else:
        # print("VEHICLE - OUT")
        # print(f'CURRENT FLOOR : {current_floor_position} && NEXT FLOOR {next_floor_position}')            
        if current_slot == 0:
            if current_vehicle_total > 0 and current_vehicle_total <= current_max_slot:
                # print("CURRENT OUT 1")
                current_slot_update = current_slot + 1
                db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

                current_vehicle_total_update = current_vehicle_total - 1
                db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)

                if next_floor_position > 1:
                    if next_slot == 0:
                        # print("NEXT OUT 1")
                        if next_vehicle_total >= next_max_slot:
                            next_vehicle_total_update = next_vehicle_total_update + 1
                            db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)
                    elif next_slot > 0 and next_slot <= next_max_slot:
                        # print("NEXT OUT 2")
                        next_slot_update = next_slot - 1
                        db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

                        next_vehicle_total_update = next_vehicle_total_update + 1
                        db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)

            elif current_vehicle_total > current_max_slot:
                # print("CURRENT OUT 2")
                current_slot_update = current_slot
                db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

                current_vehicle_total_update = current_vehicle_total + 1
                db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)

                if next_floor_position > 1:
                    if next_slot == 0:
                        if next_vehicle_total > next_max_slot:
                            next_vehicle_total_update = next_vehicle_total_update + 1
                            db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)
                    elif next_slot > 0 and next_slot <= next_max_slot:
                        next_slot_update = next_slot - 1
                        db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

                        next_vehicle_total_update = next_vehicle_total_update + 1
                        db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)


        elif current_slot > 0 and current_slot <= current_max_slot:
            if current_slot == 18:
                # print("CURRENT OUT 3")
                current_slot_update = current_slot
                db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)                    
            else:
                # print("CURRENT OUT 4")
                current_slot_update = current_slot + 1
                db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

            if current_vehicle_total == 0:
                current_vehicle_total_update = current_vehicle_total
                db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)
            else:
                current_vehicle_total_update = current_vehicle_total - 1
                db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)

            if next_floor_position > 1:
                if next_slot == 0:
                    # print("NEXT OUT 3")
                    if next_vehicle_total > next_max_slot:
                        next_slot_update = next_slot
                        db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

                        next_vehicle_total_update = next_vehicle_total + 1
                        db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)
                elif next_slot > 0 and next_slot <= next_max_slot:
                    # print("NEXT OUT 4")
                    next_slot_update = next_slot - 1
                    db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

                    next_vehicle_total_update = next_vehicle_total + 1
                    db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)
                elif next_slot > next_max_slot:
                    # print("NEXT OUT 5")
                    next_slot_update = next_slot
                    db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

                    next_vehicle_total_update = next_vehicle_total + 1
                    db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)

        # print("current_slot_update: ", current_slot_update)
        # print("next_vehicle_total_update: ", next_vehicle_total_update)

        # print(f'parking_spaces_available: {total_slot}, car_total: {vehicle_total}')

    matrix_update = MatrixController(arduino_idx, max_car=current_max_slot, total_car=current_slot_update)
    available_space = matrix_update.get_total()

    print(f"PLAT_NO : {plate_no}, AVAILABLE PARKING SPACES : {available_space}, "
        f"STATUS : {'TAMBAH' if not car_direction else 'KURANG'}, "
        f"VEHICLE_TOTAL: {current_vehicle_total_update}, FLOOR : {floor_id}, "
        f"CAMERA : {cam_id}, TOTAL_FRAME: {len(container_plate_no)}")

    db_vehicle_history.create_vehicle_history_record(
        plate_no=plate_no,
        floor_id=floor_id,
        camera=cam_id
    )

    char = "H" if plate_no_is_registered else "M"
    matrix_text = f"{plate_no},{char};"
    # self.matrix_text.write_arduino(matrix_text)

    if not db_plate.check_exist_plat(plate_no):
        plate_no_is_registered = False
        logger.write(
            f"WARNING THERE IS NO PLATE IN DATABASE!!! text: {plate_no}, status: {car_direction}",
            logger.WARNING
        )

    return current_max_slot, current_slot_update, current_vehicle_total_update