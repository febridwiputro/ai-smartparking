import threading
import time
import multiprocessing as mp
import cv2
from sympy.codegen import Print
from ultralytics import YOLO
from queue import Queue, Empty
from utils.CustomCv2 import CameraV1

def resize_image(image, max_width, max_height):
    """Resize image while maintaining aspect ratio."""
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scale = max_width / width if width > height else max_height / height
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return image

class VehicleDetector:
    def __init__(self, weight_path, frame_queue, result_queue):
        self.weight_path = weight_path
        self.model = None
        self.stopped = threading.Event()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.vehicle_thread = None
        self.current_result_frame = None

    def start(self):
        """Start the vehicle detection thread."""
        print("[Thread] Starting vehicle detection thread...")
        self.vehicle_thread = threading.Thread(target=self.detect_vehicle_work_thread, daemon=True)
        self.vehicle_thread.start()

    def stop(self):
        """Stop the detection thread gracefully."""
        self.stopped.set()
        if self.vehicle_thread is not None:
            self.vehicle_thread.join()

    def detect_vehicle_work_thread(self):
        """Thread that handles vehicle detection."""
        try:
            self.model = YOLO(self.weight_path)
            print("[Thread] Model loaded successfully.")
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")
            return

        while not self.stopped.is_set():
            try:
                frame_bundle = self.frame_queue.get(timeout=1)  # Timeout ensures the thread won't hang.
                frame = frame_bundle.get("frame")
                print(frame)
                if frame is None or frame.size == 0:
                    print("[Warning] Empty frame received.")
                    continue

                # Resize the frame and run model prediction
                frame = resize_image(frame, 640, 640)
                print(frame.shape)
                results = self.model.predict(frame, device="cuda:0", verbose=False, stream=False)
                self.current_result_frame = results[0].plot()  # Store the result for display
                print(results)

            except Empty:
                continue  # No frames in the queue, retry
            except Exception as e:
                print(f"[Error] Exception in detection thread: {e}")

VIDEO_SOURCE_PC = [
    r"C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4",
    r"C:\Users\DOT\Documents\febri\github\combined_video_out.mp4",
    r"C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4",
    r"C:\Users\DOT\Documents\febri\github\combined_video_out.mp4",
    r"C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4",
    r"C:\Users\DOT\Documents\febri\github\combined_video_out.mp4",
    r"C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4",
    r"C:\Users\DOT\Documents\febri\github\combined_video_out.mp4",
]

def main():
    """Main function to start video capture and detection."""
    result_queue = mp.Queue()
    frame_queues = [Queue(maxsize=5) for _ in VIDEO_SOURCE_PC]
    detectors = [VehicleDetector(r"C:\Users\DOT\Documents\febri\weights\yolov8n.pt", frame_queue, result_queue)
                 for frame_queue in frame_queues]

    # Start all detectors
    for detector in detectors:
        detector.start()

    caps = [CameraV1(address=video, is_video=True) for video in VIDEO_SOURCE_PC]
    for cap in caps:
        cap.start()

    print("[Main] All models loaded. Starting detection...")

    try:
        while True:
            # Read frames from all video sources
            for i, cap in enumerate(caps):
                num, frame = cap.read()
                if frame is not None:
                    # cv2.imshow(f"Frame {i}", frame)
                    if not frame_queues[i].full():
                        frame_queues[i].put({"frame": frame})

            # Display current detection results
            for i, detector in enumerate(detectors):
                if detector.current_result_frame is not None:
                    cv2.imshow(f"Result Frame {i}", detector.current_result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Main] Exit key 'q' pressed. Stopping...")
                break

    finally:
        print("[Main] Releasing resources...")
        for cap in caps:
            cap.release()

        for detector in detectors:
            detector.stop()

if __name__ == "__main__":
    main()
