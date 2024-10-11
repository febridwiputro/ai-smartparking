import os, sys
import multiprocessing as mp
from ultralytics import YOLO


from src.config.config import config

def image_restoration(stopped, patch_queue):
    pass

def text_detection(stopped, patch_queue):
    pass

def character_recognition(stopped, patch_queue):
    pass


class VehicleDetector:
    def __init__(self):
        self.stopped = mp.Event()
        self.vehicle_frame_queue = None
        self.vehicle_detected_event = mp.Event()
        self.plate_frame_queue = None
        self.frame_detected_event = mp.Event()

    def start(self):
        pass

    def stop(self):
        pass

    def process(self):
        pass

    def vehicle_detection(self):
        print(f"[Process {os.getpid()}] Start detecting the vehicle...")

        model_path = config.MODEL_PATH

        model = YOLO(model_path, task="detect")
        model.predict(verbose=False, device="cuda:0")

    def plate_detection(self):
        print(f"[Process {os.getpid()}] Start detecting the plate...")

        model_path = config.MODEL_PATH_PLAT_v2

        model = YOLO(model_path, task="detect")
        model.predict(verbose=False, device="cuda:0")

    def send_plate_no(self):
        pass