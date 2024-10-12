import cv2
import numpy as np

from src.config.config import config
from src.config.logger import logger



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