import threading
import time
from typing import List

import Levenshtein as lev
from filterpy.kalman import KalmanFilter
from scipy.ndimage import interpolation as inter
import numpy as np
import cv2

dict_to_bp = {i for i in []}


def calculate_error_rate(text, real_text):
    # total_diff = (distance(text, real_text))
    similar_word = [i for i in real_text if i not in text]
    # correct_len = len(similar_word) - total_diff
    return similar_word
    # - CER  = T/(T+C) * 100% (rumus CER )


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

# def is_point_under_line(start_point, end_point, point):
#     x1, y1 = start_point
#     x2, y2 = end_point
#     x, y = point
#
#     # Calculate A, B, and C
#     A = y2 - y1
#     B = x1 - x2
#     C = (x2 * y1) - (x1 * y2)
#
#     # Evaluate the line equation for the point
#     result = A * x + B * y + C
#     return result < 0


def auto_rescalling(target: tuple, image):
    size = image.shape[:2]
    if target[0] > size[0] or target[1] > size[1]:
        if size[0] > size[1]:
            scale = target[0] / size[0]
        else:
            scale = target[1] / size[1]
        image = cv2.resize(image, (int(size[1] * scale), int(size[0] * scale)))
    return image


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, ang):
        data = inter.rotate(arr, ang, reshape=False, order=0)
        hist = np.sum(data, axis=1, dtype=float)
        scr = np.sum((hist[1:] - hist[:-1]) ** 2, dtype=float)
        return hist, scr
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)
    
    best_angle = angles[scores.index(max(scores))]
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, float(best_angle), 1.0)
    corrected = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    
    return best_angle, corrected


def average(lst):
    list_floatt = [float(i) for i in lst]
    return round(sum(list_floatt) / len(lst), 3)


def most_freq(lst):
    return max(set(lst), key=lst.count) if lst else ""


def calculate_moving_average(curve, radius):
    # Calculate the moving average of a curve using a given radius
    window_size = 2 * radius + 1
    kernel = np.ones(window_size) / window_size
    curve_padded = np.lib.pad(curve, (radius, radius), 'edge')
    smoothed_curve = np.convolve(curve_padded, kernel, mode='same')
    smoothed_curve = smoothed_curve[radius:-radius]
    return smoothed_curve


def motion_detection(frame, avg):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if avg is None:
        avg = gray.copy().astype("float")
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    thresh_frame = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = any(cv2.contourArea(contour) > 1000 for contour in cnts)
    if motion_detected:
        for contour in cnts:
            if cv2.contourArea(contour) < 1000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return avg, motion_detected


def calculate_intersection_area(box_points, lot_polygon):
    box_polygon = cv2.convexHull(np.array(box_points))
    lot_polygon = cv2.convexHull(np.array(lot_polygon))
    
    ret, intersection_polygon = cv2.intersectConvexConvex(box_polygon, lot_polygon)
    if ret:
        return cv2.contourArea(intersection_polygon)
    return 0


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


def line_circle_intersection(line_start, line_end, circle_center, radius):
    (x1, y1) = line_start
    (x2, y2) = line_end
    (cx, cy) = circle_center
    
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    distance = abs(A * cx + B * cy + C) / np.sqrt(A ** 2 + B ** 2)
    
    if distance <= radius:
        line_vec = np.array([x2 - x1, y2 - y1])
        center_vec = np.array([cx - x1, cy - y1])
        
        projection_length = np.dot(center_vec, line_vec) / np.linalg.norm(line_vec)
        projection = projection_length * (line_vec / np.linalg.norm(line_vec))
        intersection_point = np.array([x1, y1]) + projection
        
        if 0 <= projection_length <= np.linalg.norm(line_vec):
            if (y1 < cy < y2) or (y2 < cy < y1):
                return True
    
    return False


def get_centroids(results, line_above=True) -> list[list[float | int]]:
    box = results[0].boxes.xyxy.cpu().tolist()
    detection = []
    for result in box:
        x1, y1, x2, y2 = map(int, result)
        centroid_x = (x1 + x2) / 2 if line_above else x1
        centroid_y = y2 if line_above else (y1+y2)/2
        detection.append([centroid_x, centroid_y])
    return detection


def is_point_in_polygon(polygon_points, point):
    polygon_points = np.array(polygon_points, np.int32)
    polygon_points = polygon_points.reshape((-1, 1, 2))
    distance = cv2.pointPolygonTest(polygon_points, point, False)
    return distance >= 0


def do_intersect(p1, q1, p2, q2):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2
    
    def on_segment(p, q, r):
        return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
    
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    if o1 != o2 and o3 != o4:
        return True
    
    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q1, q2):
        return True
    if o3 == 0 and on_segment(p2, q2, p1):
        return True
    if o4 == 0 and on_segment(p2, q2, q1):
        return True
    
    return False


def rect_line_intersect(rect, line):
    x1, y1, x2, y2 = rect
    (x3, y3), (x4, y4) = line
    
    rect_edges = [((x1, y1), (x2, y1)),
                  ((x2, y1), (x2, y2)),
                  ((x2, y2), (x1, y2)),
                  ((x1, y2), (x1, y1))]
    
    for edge in rect_edges:
        if do_intersect(edge[0], edge[1], (x3, y3), (x4, y4)):
            return True
    
    return False


def filtering_score(text: list):
    teks = [i[0] for i in text if i[0] != ""]
    score = [i[1] for i in text if i[0] != ""]
    most_teks = most_freq(teks)
    highest_score = max(score) if score else 0
    index_score = score.index(highest_score) if score else 0
    highest_score_teks = teks[index_score] if teks else ""
    
    if most_teks != highest_score_teks:
        return most_teks
    elif most_teks == highest_score_teks and most_teks != "":
        print("score sama dengan most teks", highest_score_teks)
        return highest_score_teks


def split_frame(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break


def find_closest_strings_dict(target, strings):
    distances = np.array([lev.distance(target, s) for s in strings])
    min_distance = np.min(distances)
    min_indices = np.where(distances == min_distance)[0]
    closest_strings_dict = {strings[i]: distances[i] for i in min_indices}
    return closest_strings_dict


def is_up(prev_y, curr_y, threshold=10):
    delta_y = curr_y - prev_y
    if abs(delta_y) < threshold:
        return None  # Tidak ada gerakan yang signifikan
    return delta_y < 0

def get_movement_direction(centroids):
    # Initialize Kalman Filter
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0, 0, 0, 0])  # initial state (position and velocity)
    kf.P *= 1000  # initial uncertainty
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])  # state transition matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])  # measurement function
    kf.R = np.array([[10, 0],
                     [0, 10]])  # measurement noise
    kf.Q = np.eye(4)  # process noise
    
    if not centroids:
        return "stationary"  # No centroids available, assume stationary
    
    # Update Kalman Filter with new measurement
    kf.predict()
    kf.update(np.array(centroids[0]))
    
    # Extract predicted velocity
    velocity = kf.x[2:]
    
    # Determine movement direction
    if velocity[1] > 0:
        return "down"
    elif velocity[1] < 0:
        return "up"
    else:
        return "stationary"
    
def crop_polygon(image, points):
    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1, 1, 2))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], (255,255,255))
    result = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(points)
    cropped_image = result[y:y + h, x:x + w]
    return cropped_image


if __name__ == "__main__":
    # rect = (0, 0, 10, 10)  # (x1, y1, x2, y2)
    # line = ((5, -5), (5, 5))  # Line segment from (5, -5) to (5, 15)
    # print(rect_line_intersect(rect, line))  # Output: True
    
    # find closest_string
    strings = ['BP1030TZ', 'BP3192OJ', 'BP1668OD', 'BP1773IO', 'BP1552OD', 'BP1628RF', 'BP1012OZ', 'BP7046VE', 'BP30VV',
               'BP1553OD', 'BP1562RF', 'BP1062KZ', 'BP1062MF', 'BP1062SC', 'BP1332OD', 'BP1030MZ', 'BP9290HI',
               'BP1080FZ', 'BP1895OD', 'BP8407DQ', 'BP1623QO', 'BP1776IO', 'BP1062M', 'BP1030IZ', 'BP1558OD',
               'BP1768QE', 'BP1808QC', 'BP1485RE', 'BP1021OZ', 'BP1080HZ', 'BP1715IO', 'BP7047VE', 'BP7436DD',
               'BP2721JG', 'BP7250DD', 'BP1778OD', 'BP1062LB', 'BP2722JG', 'BP1663IO', 'BP1457GO', 'BP1320AM',
               'BP1807QC', 'BP1586YY', 'BP7420DD', 'BP2720JG', 'BP1062OZ', 'BP7051ZE', 'BP1125OZ', 'BP1559IZ',
               'BP1897ME', 'BP1505IZ', 'BP1460GO', 'BP1458GO', 'BP1018CF', 'BP1767QE', 'BP1062RC', 'BP1030ZT',
               'BP1030LC', 'BP3265FK', 'BP7579DD', 'BP30VP', 'BP1805QC', 'BP8892DQ', 'BP1062DJ', 'BP1626QO', 'BP7630ZE',
               'BP1559OD', 'BP1030PZX', 'BP1050RE', 'BP1080LZ', 'BP1015CF', 'BP9240HI', 'BP1067CF', 'BP1557OD',
               'BP1563RF', 'BP1556OD', 'BP7155ZB', 'BP8AB', 'BP7592DD', 'BP1062GN', 'BP1012CF', 'BP1801QC', 'BP7156ZB',
               'BP1670IZ', 'BP1132OZ', 'BP1621QO', 'BP1318AM', 'BP9436HI', 'BP7153ZB', 'BP1626OQ', 'BP1062PN',
               'BP1731YF', 'BP1620QO', 'BP1062SN', 'BP1892OD', 'BP1062ZS', 'BP1030ZQ', 'BP1030KZ', 'BP1802QC',
               'BP7029ZE', 'BP1013CF', 'BP8392EH', 'BP1665OD', 'BP1030SY', 'BP2033OM', 'BP1080EZ', 'BP7052ZE',
               'BP1491QO', 'BP1062KK', 'BP1319AM', 'BP1062LZ', 'BP1265GO', 'BP1564RF', 'BP6916MJ', 'BP1062PZ',
               'BP1065CF', 'BP1176QO', 'BP1080DZ', 'BP1030ZJ', 'BP1080KZ', 'BP1669OD', 'BP1030DZ', 'BP1383IZ',
               'BP1062HF', 'BP1030QZ', 'BP1397IZ', 'BP1016CF', 'BP1030WZ', 'BP1809QC', 'BP1051RE', 'BP1634OQ',
               'BP1459GO', 'BP1803QC', 'BP1013OZ', 'BP1806QC', 'BP1536MF', 'BP1281QO', 'BP1662OD', 'BP1627QO',
               'BP2043OM', 'BP1017CF', 'BP7248DD', 'BP1062EZ', 'BP1667OD', 'BP1062HN', 'BP1871IO', 'BP2042OM']
    target = 'BP1760QE'
    end = time.perf_counter()
    closest_string = find_closest_strings_dict(target, strings)
    print(closest_string)  # print closest fitting
    print(f"Function time = {time.perf_counter() - end}")  # 3.830005880445242e-05
    
    # Example 1: Object moving up
    prev_y_up = 15
    curr_y_up = 18
    direction_up = is_up(prev_y_up, curr_y_up)
    
    print("Moving up" if direction_up else "Moving down")
    
    # Example 2: Object moving down
    prev_y_down = 18
    curr_y_down = 15
    direction_down = is_up(prev_y_down, curr_y_down)
    
    print("Moving up" if direction_down else "Moving down")
    # point = (25 , 50)
    # point_line_1 = (50, 25)
    # point_lint_2 = (50, 100)
    
    
    # print(point_position(point_line_1, point_lint_2, point, inverse=False))
