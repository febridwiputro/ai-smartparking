import os
import cv2
import numpy as np
import multiprocessing as mp
from ultralytics import YOLO
from norfair import Detection
from utils.centroid_tracking import CentroidTracker
from src.config.config import config

def preprocess(image: np.ndarray) -> np.ndarray:
    """Preprocess the image for YOLO model."""
    return image  # Skip grayscale conversion; YOLO expects color images

def draw_boxes(frame, results):
    """Draw bounding boxes and centroids."""
    for box in results.boxes:
        cls_id = int(box.cls.cpu().numpy())
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
        color = (255, 255, 255)  # White color for bounding box
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot
    
    return frame

def _predict_yolo(stopped: mp.Event, frame_queue: mp.Queue, result_queue: mp.Queue, model_built_event: mp.Event, yolo_model: YOLO):
    """YOLO object detection process."""
    print(f"[Process {os.getpid()}] Start detecting objects with YOLO...")
    model_built_event.set()

    while not stopped.is_set():
        try:
            frame = frame_queue.get()

            if frame is None:
                continue

            # Preprocess and predict
            results = yolo_model.predict(frame, conf=0.25, device="cuda:0", verbose=False)
            
            # If no results, skip processing
            if results and hasattr(results[0], 'boxes'):
                result_queue.put(results)
            else:
                print("No bounding boxes detected.")

        except Exception as e:
            print(f"Error in _predict_yolo: {e}")

    print(f"[Process {os.getpid()}] Stop detecting objects...")

def _track_centroids(stopped: mp.Event, centroid_queue: mp.Queue, tracking_queue: mp.Queue, centroid_tracking: CentroidTracker, tracking_built_event: mp.Event):
    """Track centroids in a separate process."""
    print(f"[Process {os.getpid()}] Start tracking centroids with CentroidTracker...")
    tracking_built_event.set()

    while not stopped.is_set():
        try:
            centroids = centroid_queue.get()
            if centroids is None:
                continue

            tracked_objects = centroid_tracking.update(np.array(centroids))

            coordinates = [list(obj.flatten()) for obj in tracked_objects.values()]
            ids = [id for id in tracked_objects.keys()]
            tracking_results = {"coordinates": coordinates, "ids": ids}

            tracking_queue.put(tracking_results)

        except Exception as e:
            print(f"Error in _track_centroids: {e}")

    print(f"[Process {os.getpid()}] Stop tracking centroids...")

class VehicleDetectorMP:
    """Class to handle vehicle detection using multiprocessing."""
    def __init__(self, yolo_model):
        self.yolo_model = yolo_model
        self.frame_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.tracking_queue = mp.Queue()
        self.stopped = mp.Event()
        self.model_built_event = mp.Event()
        self.tracking_built_event = mp.Event()
        self.centroid_queue = mp.Queue()

        self.centroid_tracking = CentroidTracker(maxDisappeared=75)

        self.detector_process = None
        self.tracking_process = None

    def start_multiprocessing(self):
        """Start the object detection and tracking processes."""
        self.detector_process = mp.Process(
            target=_predict_yolo,
            args=(self.stopped, self.frame_queue, self.result_queue, self.model_built_event, self.yolo_model)
        )
        self.detector_process.start()

        self.tracking_process = mp.Process(
            target=_track_centroids,
            args=(self.stopped, self.centroid_queue, self.tracking_queue, self.centroid_tracking, self.tracking_built_event)
        )
        self.tracking_process.start()

    def stop_multiprocessing(self):
        """Stop all multiprocessing."""
        self.stopped.set()
        self.frame_queue.put(None)
        self.result_queue.put(None)
        self.tracking_queue.put(None)

        if self.detector_process is not None:
            self.detector_process.join()

        if self.tracking_process is not None:
            self.tracking_process.join()

    def process_frame(self, frame):
        """Send frame to detector process."""
        if self.is_model_ready():
            self.frame_queue.put(frame)

    def is_model_ready(self):
        """Check if YOLO model is ready."""
        return self.model_built_event.is_set()

    def is_tracking_ready(self):
        """Check if tracking is ready."""
        return self.tracking_built_event.is_set()

    def get_tracking_centroid(self, centroids):
        """Get tracking results from the tracking process."""
        if self.is_tracking_ready():
            self.centroid_queue.put(centroids)
        else:
            print("Tracking process is not ready yet.")
            return [], []

        if not self.tracking_queue.empty():
            try:
                tracking_results = self.tracking_queue.get(timeout=1)
                return tracking_results["coordinates"], tracking_results["ids"]
            except Exception as e:
                print(f"Error getting tracking results: {e}")
                return [], []

        return [], []

# import os
# import cv2
# import numpy as np
# import multiprocessing as mp
# from ultralytics import YOLO
# from norfair import Detection
# from utils.centroid_tracking import CentroidTracker
# from src.config.config import config

# def preprocess(image: np.ndarray) -> np.ndarray:
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
#     return image_bgr

# def draw_boxes(frame, results):
#     for box in results.boxes:
#         cls_id = int(box.cls.cpu().numpy())
#         x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
#         # label = CLASS_NAMES[cls_id]
#         color = (255, 255, 255)  # Green color for bounding box
        
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
#         # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
#         center_x = (x1 + x2) // 2
#         center_y = (y1 + y2) // 2
#         cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot
    
#     return frame

# def _predict_yolo(stopped: mp.Event, frame_queue: mp.Queue, result_queue: mp.Queue, model_built_event: mp.Event, yolo_model: YOLO):
#     print(f"[Process {os.getpid()}] Start detecting objects with YOLO...")
    
#     model_built_event.set()

#     while not stopped.is_set():
#         try:
#             frame = frame_queue.get()

#             if frame is None:
#                 continue

#             # preprocessed_image = preprocess(frame)
#             results = yolo_model.predict(frame, conf=0.25, device="cuda:0", verbose=False, classes=config.CLASS_NAMES)

#             for result in results:
#                 draw_boxes(frame=frame, results=result)

#             # Kirim hasil ke queue
#             result_queue.put(results)
#         except Exception as e:
#             print(f"Error in _predict_yolo: {e}")

#     print(f"[Process {os.getpid()}] Stop detecting objects...")

# # Object tracking function running in a separate process
# def _track_centroids(stopped: mp.Event, centroid_queue: mp.Queue, tracking_queue: mp.Queue, centroid_tracking: CentroidTracker, tracking_built_event: mp.Event):
#     print(f"[Process {os.getpid()}] Start tracking centroids with CentroidTracker...")
#     tracking_built_event.set()  # Notify that tracker is ready

#     while not stopped.is_set():
#         try:
#             centroids = centroid_queue.get()
#             if centroids is None:
#                 continue

#             # Update tracking using CentroidTracker with input centroids
#             tracked_objects = centroid_tracking.update(np.array(centroids))

#             # Convert tracking results into the correct format
#             coordinates = [list(obj.flatten()) for obj in tracked_objects.values()]
#             ids = [id for id in tracked_objects.keys()]
#             tracking_results = {"coordinates": coordinates, "ids": ids}

#             # Send tracking results to tracking_queue
#             tracking_queue.put(tracking_results)

#         except Exception as e:
#             print(f"Error in _track_centroids: {e}")

#     print(f"[Process {os.getpid()}] Stop tracking centroids...")


# class VehicleDetectorMP:
#     def __init__(self, yolo_model):
#         self.yolo_model = yolo_model  # Set YOLO model
#         self.frame_queue = mp.Queue()
#         self.result_queue = mp.Queue()
#         self.tracking_queue = mp.Queue()
#         self.stopped = mp.Event()
#         self.model_built_event = mp.Event()
#         self.tracking_built_event = mp.Event()
#         self.centroid_queue = mp.Queue()

#         # Initialize CentroidTracker for tracking centroids
#         self.centroid_tracking = CentroidTracker(maxDisappeared=75)

#         # Process placeholders
#         self.detector_process = None
#         self.tracking_process = None

#     def start_multiprocessing(self):
#         self.detector_process = mp.Process(
#             target=_predict_yolo,
#             args=(self.stopped, self.frame_queue, self.result_queue, self.model_built_event, self.yolo_model)
#         )
#         self.detector_process.start()

#         self.tracking_process = mp.Process(
#             target=_track_centroids,
#             args=(self.stopped, self.centroid_queue, self.tracking_queue, self.centroid_tracking, self.tracking_built_event)
#         )
#         self.tracking_process.start()

#     def stop_multiprocessing(self):
#         self.stopped.set()
#         self.frame_queue.put(None)
#         self.result_queue.put(None)
#         self.tracking_queue.put(None)

#         if self.detector_process is not None:
#             self.detector_process.join()

#         if self.tracking_process is not None:
#             self.tracking_process.join()

#     def process_frame(self, frame):
#         # Make sure model is ready before processing frames
#         if self.is_model_ready():
#             self.frame_queue.put(frame)

#     def is_model_ready(self):
#         return self.model_built_event.is_set()

#     def is_tracking_ready(self):
#         return self.tracking_built_event.is_set()

#     def get_tracking_centroid(self, centroids):
#         # Kirim centroid ke tracking queue untuk diproses
#         if self.is_tracking_ready():
#             self.centroid_queue.put(centroids)
#         else:
#             print("Tracking process is not ready yet.")
#             return [], []

#         # Ambil hasil pelacakan dari tracking_queue
#         if not self.tracking_queue.empty():
#             try:
#                 tracking_results = self.tracking_queue.get(timeout=1)
#                 print("Coordinates: ", tracking_results["coordinates"], "IDs: ", tracking_results["ids"])
#                 return tracking_results["coordinates"], tracking_results["ids"]
#             except Exception as e:
#                 print(f"Error getting tracking results: {e}")
#                 return [], []
        
#         return [], []