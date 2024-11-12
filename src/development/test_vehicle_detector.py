import threading
import time
import multiprocessing as mp

import cv2
from ultralytics import YOLO

from utils.CustomCv2 import CameraV1

def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scale = max_width / width if width > height else max_height / height
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return image

class VehicleDetector:
    def __init__(self, weight_path, result_queue=None):
        self.weight_path = weight_path
        self.model = None
        self.current_frame_bundle = {}
        self.stopped = threading.Event()
        self.model_built_event = threading.Event()
        self.vehicle_thread = None
        self.result_queue = result_queue
        
        self.current_result_frame = None
        self.frame_lock = threading.Lock()
    
    
    def start(self):
        print("[Thread] Starting vehicle detection thread...")
        self.vehicle_thread = threading.Thread(target=self.detect_vehicle_work_thread)
        self.vehicle_thread.start()
    
    def stop(self):
        self.stopped.set()
        if self.vehicle_thread is not None:
            self.vehicle_thread.join()
    
    def is_model_built(self):
        return self.model_built_event.is_set()

    def process_frame(self, frame_bundle: dict):
        self.current_frame_bundle = frame_bundle.copy()

    def detect_vehicle_work_thread(self):
        model = YOLO(self.weight_path)
        self.model_built_event.set()

        while True:
            if self.stopped.is_set():
                break
            
            try:
                if self.current_frame_bundle is None or len(self.current_frame_bundle) == 0:
                    print("Empty or invalid frame received.")
                    time.sleep(0.1)
                    continue
    
                frame_bundle = self.current_frame_bundle.copy()
    
                frame = frame_bundle.get("frame")
                print(frame.shape)
                if frame is None or frame.size == 0:
                    print("Empty or invalid frame received.")
                    time.sleep(0.1)
                    continue
                
                frame = resize_image(frame, 640, 640)
                results = model.predict(frame, device="cuda:0", verbose=True, stream=False)
                
                
                # self.current_result_frame = results[0].plot()


                # boxes = [result.boxes.xyxyresults]
            except Exception as e:
                print("Error at detect_vehicle_work_thread", e)
                continue


VIDEO_SOURCE_PC = [
    fr"C:\Users\DOT\Documents\febri\video\sequence\LT_5_IN.mp4",
    fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4',
    fr'C:\Users\DOT\Documents\febri\github\combined_video_out.mp4',
    fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4',
    # fr'C:\Users\DOT\Documents\febri\github\combined_video_out.mp4',
    # fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4',
    # fr'C:\Users\DOT\Documents\febri\github\combined_video_out.mp4',
    # fr'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4',
    # fr'C:\Users\DOT\Documents\febri\github\combined_video_out.mp4'
]

def main():
    result_queue = mp.Queue()
    weight_path = r"C:\Users\DOT\Documents\febri\weights\yolo11n.pt"
    
    num_processes = len(VIDEO_SOURCE_PC)
    detectors = [VehicleDetector(weight_path=weight_path, result_queue=result_queue) for _ in range(num_processes)]
    for detector in detectors:
        detector.start()
    
    frames = [None for _ in range(num_processes)]
    caps = [CameraV1(address=video, is_video=True) for video in VIDEO_SOURCE_PC]
    for cap in caps:
        cap.start()
    
    # while not all([m.is_model_built() for m in detectors]):
    #     time.sleep(0.1)
    #     print("Loading detection models...")
    
    print("All models are loaded. Starting detection...")

    while True:
        for i, cap in enumerate(caps):
            num, frames[i] = cap.read()
            print("detectors[i].process_frame")
            detectors[i].process_frame({"frame": frames[i].copy()})
        
        for i, frame in enumerate(frames):
            if frame is not None:
                frame = resize_image(frame, 640, 640)
                cv2.imshow(f"Frame {i}", frame)

        for i, detector in enumerate(detectors):
            if detector.current_result_frame is not None:
                cv2.imshow(f"Frame Result {i}", detector.current_result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit key 'q' pressed. Stopping...")
            break
    
    for cap in caps:
        cap.release()
    
    for detector in detectors:
        detector.stop()

    # vehicle_detector.start()
    # for video in VIDEO_SOURCE_PC:
    #     print(f"Processing video: {video}")
    #     vehicle_detector.process_frame({"frame": video})
    #     time.sleep(5)
    # vehicle_detector.stop()


if  __name__ == "__main__":
    main()