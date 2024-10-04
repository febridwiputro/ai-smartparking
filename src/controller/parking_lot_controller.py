import time
import cv2
import numpy as np
from shapely.geometry import Point, Polygon

class ParkingLot:
    def __init__(self, lot_id, polygon):
        self.polygon = np.array(polygon, np.int32)
        self.occupied = False
        self.occupied_start_time = None
        self.empty_start_time = None
        self.pos = False
        self.id = lot_id
        self.initialized = False  # Flag to indicate if initial status has been checked

    def update_occupancy(self, frame, result):
        polygon_shape = Polygon(self.polygon)

        if not self.initialized:
            # Check initial status without timer
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_point = Point(center_x, center_y)
                if polygon_shape.contains(center_point):
                    self.occupied = True
                    self.pos = True
                    self.initialized = True
                    break
            if not self.occupied:
                self.initialized = True

        if self.occupied:
            still_occupied = False
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_point = Point(center_x, center_y)
                if polygon_shape.contains(center_point):
                    still_occupied = True
                    self.pos = True
                    self.empty_start_time = None  # Reset empty start time if a car is detected
                    break

            if not still_occupied:
                if self.empty_start_time is None:
                    self.empty_start_time = time.time()
                elif time.time() - self.empty_start_time >= 1:
                    self.occupied = False
                    self.occupied_start_time = None
                    self.pos = False
        else:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_point = Point(center_x, center_y)
                if polygon_shape.contains(center_point):
                    if self.occupied_start_time is None:
                        self.occupied_start_time = time.time()
                    elif time.time() - self.occupied_start_time >= 1:
                        self.occupied = True
                        self.pos = True
                        self.empty_start_time = None  # Reset empty start time if a car is detected
                    break

        # color = (0, 0, 255) if self.occupied else (0, 255, 0)
        # cv2.polylines(frame, [self.polygon.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=2)
        # status = 'Occupied' if self.occupied else 'Empty'
        # text_pos = tuple(self.polygon[0])
        # cv2.putText(frame, str(self.id), (text_pos[0], text_pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame
