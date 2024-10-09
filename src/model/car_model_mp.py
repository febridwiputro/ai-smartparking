import numpy as np
import cv2
import multiprocessing as mp
from ultralytics import YOLO
from norfair import Detection, Tracker
from utils.centroid_tracking import CentroidTracker
from src.config.config import config

# YOLO prediction function running in a separate process
def _predict_yolo(stopped: mp.Event, frame_queue: mp.Queue, result_queue: mp.Queue, model_built_event: mp.Event):
    print(f"[Process {os.getpid()}] Start detecting objects with YOLO...")
    model = YOLO(config.MODEL_PATH)  # Load YOLO model
    model_built_event.set()  # Notify that model is ready

    while not stopped.is_set():
        try:
            frame = frame_queue.get()

            if frame is None:
                continue

            # Preprocess and predict
            frame_resized = cv2.resize(frame, (640, 640))  # Resize the frame for YOLO
            results = model.predict(frame_resized, conf=0.25, device="cuda:0", verbose=False)[0]

            # Send results to result_queue
            result_queue.put(results)
        except Exception as e:
            print(f"Error in _predict_yolo: {e}")

    print(f"[Process {os.getpid()}] Stop detecting objects...")

# Object tracking function running in a separate process
def _track_yolo(stopped: mp.Event, frame_queue: mp.Queue, result_queue: mp.Queue, tracking_event: mp.Event):
    print(f"[Process {os.getpid()}] Start tracking objects...")
    tracker = Tracker(distance_function='euclidean', distance_threshold=10)  # Initialize tracker
    centroid_tracker = CentroidTracker(maxDisappeared=75)  # Centroid-based tracker
    tracking_event.set()  # Notify that tracker is ready

    while not stopped.is_set():
        try:
            frame = frame_queue.get()

            if frame is None:
                continue

            # Preprocess the frame for tracking
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

            results = result_queue.get()  # Retrieve the results from YOLO
            detections = [Detection(np.array([x1, y1, x2, y2])) for (x1, y1, x2, y2) in results.boxes.xyxy.cpu().numpy()]

            # Update the tracker
            tracked_objects = tracker.update(detections)

            # Handle the tracking information (e.g., draw bounding boxes, centroids, etc.)
            for obj in tracked_objects:
                cv2.rectangle(frame, (int(obj.estimate[0]), int(obj.estimate[1])), 
                                     (int(obj.estimate[2]), int(obj.estimate[3])), (255, 0, 0), 2)

        except Exception as e:
            print(f"Error in _track_yolo: {e}")

    print(f"[Process {os.getpid()}] Stop tracking objects...")

# Utility function to preprocess image for YOLO
def preprocess_image(image: np.ndarray) -> np.ndarray:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    return image_bgr

# Function to draw bounding boxes around detected objects
def draw_boxes(frame, results):
    for box in results.boxes:
        cls_id = int(box.cls.cpu().numpy())
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
        color = (255, 255, 255)  # White color for bounding box
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot
        
    return frame

# Utility function to calculate distance between centroids
def calculate_distance_object(image, centroids):
    """
    Calculate distances based on the height (y-coordinate) only.

    :param image: The image on which to draw lines.
    :param centroids: List of detected centroids in the format [(x_center, y_center), ...].
    :return: List of distances and their corresponding indices, sorted by distance.
    """
    height, width = image.shape[:2]
    half_width = width / 3
    distances = []
    point1 = np.array([half_width, height - 20])

    for i, centroid in enumerate(centroids):
        point2 = np.array([half_width, centroid[1]])
        dist = np.linalg.norm(point1 - point2)
        cv2.line(image, pt1=(int(half_width), int(height)), pt2=(int(half_width), int(centroid[1])),
                 lineType=cv2.LINE_AA, thickness=2, color=(255, 255, 255))
        
        if dist >= 100:
            distances.append((i, dist))  # Store index and distance
    
    distances.sort(key=lambda x: x[1])
    return distances

# Function to get the vehicle's bounding box from results
def get_vehicle_image(image, results):
    box = results[0].boxes.xyxy.cpu().tolist()
    x1, y1, x2, y2 = map(int, box[0])
    car = image[max(y1, 0): min(y2, image.shape[0]), max(x1, 0): min(x2, image.shape[1])]
    color = (255, 255, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)
    return car

# Function to retrieve object boxes
def get_boxes(results):
    box = results[0].boxes.xyxy.cpu().tolist()
    return box[0]

class VehicleDetector:
    def __init__(self):
        self.frame_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.stopped = mp.Event()
        self.model_built_event = mp.Event()
        self.tracking_event = mp.Event()

        # Process placeholders
        self.detector_process = None
        self.tracking_process = None

    def start_multiprocessing(self):
        self.detector_process = mp.Process(
            target=_predict_yolo,
            args=(self.stopped, self.frame_queue, self.result_queue, self.model_built_event)
        )
        self.detector_process.start()

        self.tracking_process = mp.Process(
            target=_track_yolo,
            args=(self.stopped, self.frame_queue, self.result_queue, self.tracking_event)
        )
        self.tracking_process.start()

    def stop_multiprocessing(self):
        self.stopped.set()
        self.frame_queue.put(None)
        self.result_queue.put(None)

        if self.detector_process is not None:
            self.detector_process.join()

        if self.tracking_process is not None:
            self.tracking_process.join()

    def process_frame(self, frame):
        self.frame_queue.put(frame)

    def is_model_ready(self):
        return self.model_built_event.is_set()

    def is_tracker_ready(self):
        return self.tracking_event.is_set()
