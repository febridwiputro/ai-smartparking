import cv2
import numpy as np
import Levenshtein as lev

from src.config.config import config
from src.config.logger import logger
from src.Integration.service_v1.controller.plat_controller import PlatController
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.fetch_api_controller import FetchAPIController
from src.Integration.service_v1.controller.vehicle_history_controller import VehicleHistoryController

db_plate = PlatController()
db_floor = FloorController()
db_mysn = FetchAPIController()
db_vehicle_history = VehicleHistoryController()

def find_closest_strings_dict(target, strings):
    distances = np.array([lev.distance(target, s) for s in strings])
    min_distance = np.min(distances)
    min_indices = np.where(distances == min_distance)[0]
    closest_strings_dict = {strings[i]: distances[i] for i in min_indices}
    return closest_strings_dict

def most_freq(lst):
    return max(set(lst), key=lst.count) if lst else ""

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
    
    # point=polygons_point[cam_idx]

    # polygons = [point]
    # bbox = convert_decimal_to_bbox((self.height, self.width), polygons)
    # frame = crop_polygon(frame, bbox[0])

    if frame.shape[0] == () or frame.shape[1] == ():
        return "", np.array([]), np.array([])

    if floor_id == 2:
        # polygon_point = config.POLYGON_POINT_LT2_OUT
        polygon_point = config.POLYGON_POINT_LT2_IN if cam_id == "IN" else config.POLYGON_POINT_LT2_OUT
    elif floor_id == 3:
        polygon_point = config.POLYGON_POINT_LT3_IN if cam_id == "IN" else config.POLYGON_POINT_LT3_OUT
    elif floor_id == 4:
        polygon_point = config.POLYGON_POINT_LT4_IN if cam_id == "IN" else config.POLYGON_POINT_LT4_OUT
    elif floor_id == 5:
        polygon_point = config.POLYGON_POINT_LT5_IN if cam_id == "IN" else config.POLYGON_POINT_LT5_OUT
    else:
        return []

    poly_points = convert_decimal_to_bbox((height, width), polygon_point)
    
    return poly_points, frame

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

def convert_bbox_to_decimal(img_dims, polygons):
    height, width = img_dims
    normalized_polygons = []
    for pol in polygons:
        normalized_pol = []
        for bbox in pol:
            normalized_bbox = (bbox[0] / width, bbox[1] / height)
            normalized_pol.append(normalized_bbox)
        normalized_polygons.append(normalized_pol)
    return normalized_polygons

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

def parking_space_vehicle_counter(floor_id, cam_id, arduino_idx, car_direction, plate_no):
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
    if car_direction:
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

        print("VEHICLE - IN")
        print(f'CURRENT FLOOR : {current_floor_position} && PREV FLOOR {prev_floor_position}')  

        if current_slot == 0:
            print("UPDATE 0")
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
                    print("IN 1")
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
                    print("IN 2")
                    prev_slot_update = prev_slot + 1
                    # print("prev_slot_update: ", prev_slot_update)
                    # print("prev_slot_update: ", prev_slot_update)

                    db_floor.update_slot_by_id(id=prev_floor_position, new_slot=prev_slot_update)

                    prev_vehicle_total_update = prev_vehicle_total - 1
                    # print("prev_vehicle_total_update: ", prev_vehicle_total_update)
                    db_floor.update_vehicle_total_by_id(id=prev_floor_position, new_vehicle_total=prev_vehicle_total_update)

    # TURUN / KELUAR
    else:
        print("VEHICLE - OUT")
        print(f'CURRENT FLOOR : {current_floor_position} && NEXT FLOOR {next_floor_position}')            
        if current_slot == 0:
            if current_vehicle_total > 0 and current_vehicle_total <= current_max_slot:
                print("CURRENT OUT 1")
                current_slot_update = current_slot + 1
                db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)

                current_vehicle_total_update = current_vehicle_total - 1
                db_floor.update_vehicle_total_by_id(id=current_floor_position, new_vehicle_total=current_vehicle_total_update)

                if next_floor_position > 1:
                    if next_slot == 0:
                        print("NEXT OUT 1")
                        if next_vehicle_total >= next_max_slot:
                            next_vehicle_total_update = next_vehicle_total_update + 1
                            db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)
                    elif next_slot > 0 and next_slot <= next_max_slot:
                        print("NEXT OUT 2")
                        next_slot_update = next_slot - 1
                        db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

                        next_vehicle_total_update = next_vehicle_total_update + 1
                        db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)

            elif current_vehicle_total > current_max_slot:
                print("CURRENT OUT 2")
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
                print("CURRENT OUT 3")
                current_slot_update = current_slot
                db_floor.update_slot_by_id(id=current_floor_position, new_slot=current_slot_update)                    
            else:
                print("CURRENT OUT 4")
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
                    print("NEXT OUT 3")
                    if next_vehicle_total > next_max_slot:
                        next_slot_update = next_slot
                        db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

                        next_vehicle_total_update = next_vehicle_total + 1
                        db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)
                elif next_slot > 0 and next_slot <= next_max_slot:
                    print("NEXT OUT 4")
                    next_slot_update = next_slot - 1
                    db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

                    next_vehicle_total_update = next_vehicle_total + 1
                    db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)
                elif next_slot > next_max_slot:
                    print("NEXT OUT 5")
                    next_slot_update = next_slot
                    db_floor.update_slot_by_id(id=next_floor_position, new_slot=next_slot_update)

                    next_vehicle_total_update = next_vehicle_total + 1
                    db_floor.update_vehicle_total_by_id(id=next_floor_position, new_vehicle_total=next_vehicle_total_update)

        print("current_slot_update: ", current_slot_update)
        print("next_vehicle_total_update: ", next_vehicle_total_update)

    return current_max_slot, current_slot_update, current_vehicle_total_update