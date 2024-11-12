import numpy as np
import cv2
import multiprocessing as mp
from ultralytics import YOLO
from src.config.config import config

# Multiprocessing function for plate detection
def _predict_plate(stopped: mp.Event, frame_queue: mp.Queue, result_queue: mp.Queue, model_built_event: mp.Event):
    print(f"[Process {os.getpid()}] Start detecting plates...")
    model = YOLO(config.MODEL_PATH_PLAT)  # Load YOLO model for plate detection
    model_built_event.set()  # Notify that model is ready

    while not stopped.is_set():
        try:
            frame = frame_queue.get()

            if frame is None:
                continue

            # Preprocess and predict plate
            frame_resized = cv2.resize(frame, (640, 640))
            results = model.predict(frame_resized, conf=0.3, device="cuda:0", verbose=False)[0]

            # Send results to result_queue
            result_queue.put(results)
        except Exception as e:
            print(f"Error in _predict_plate: {e}")

    print(f"[Process {os.getpid()}] Stop detecting plates...")

# Plate detection class
class PlatDetector:
    def __init__(self):
        self.frame_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.stopped = mp.Event()
        self.model_built_event = mp.Event()

        # Process placeholder
        self.detector_process = None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return image_bgr

    def start_multiprocessing(self):
        self.detector_process = mp.Process(
            target=_predict_plate,
            args=(self.stopped, self.frame_queue, self.result_queue, self.model_built_event)
        )
        self.detector_process.start()

    def stop_multiprocessing(self):
        self.stopped.set()
        self.frame_queue.put(None)

        if self.detector_process is not None:
            self.detector_process.join()

    def process_frame(self, frame):
        preprocessed_frame = self.preprocess(frame)
        self.frame_queue.put(preprocessed_frame)

    def get_results(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

    def is_model_ready(self):
        return self.model_built_event.is_set()

    def draw_boxes(self, frame, results):
        for box in results.boxes.xyxy.cpu():
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    def get_plat_image(self, image, results):
        for box in results.boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = map(int, box)
            plat = image[max(y1, 0): min(y2, image.shape[0]), max(x1, 0): min(x2, image.shape[1])]
            return plat
        return np.array([])

