import os, sys
import cv2
import numpy as np
import logging
from datetime import datetime
import gc

this_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(this_path)

# print("this_path: ", this_path)

from src.config.config import config
# from src.config.logger import logger
from src.models.gan_model import GanModel

def image_restoration(stopped, model_built_event, plate_result_queue, img_restoration_result_queue):
    img_restore = ImageRestoration()
    model_built_event.set()

    while not stopped.is_set():
        try:
            plate_result = plate_result_queue.get()

            if plate_result is None or len(plate_result) == 0:
                continue

            # Extract all relevant fields from plate_result
            object_id = plate_result.get("object_id")
            plate_image = plate_result.get("frame", None)
            bg_color = plate_result.get("bg_color", None)
            floor_id = plate_result.get("floor_id", 0)
            cam_id = plate_result.get("cam_id", "")
            arduino_idx = plate_result.get("arduino_idx", None)
            car_direction = plate_result.get("car_direction", None)
            start_line = plate_result.get("start_line", None)
            end_line = plate_result.get("end_line", None)
            is_centroid_inside = plate_result.get("is_centroid_inside")

            if plate_image is None:
                continue

            empty_frame = np.empty((0, 0, 3), dtype=np.uint8)

            if not start_line and not end_line:
                result = {
                    "object_id": object_id,
                    "bg_color": bg_color,
                    "frame": empty_frame,
                    "floor_id": floor_id,
                    "cam_id": cam_id,
                    "arduino_idx": arduino_idx,
                    "car_direction": car_direction,
                    "start_line": start_line,
                    "end_line": end_line,
                    "is_centroid_inside": is_centroid_inside
                }
                img_restoration_result_queue.put(result)
                continue

            if plate_image is None:
                continue

            restored_image = img_restore.process_image(plate_image)

            result = {
                "object_id": object_id,
                "bg_color": bg_color,
                "frame": restored_image,
                "floor_id": floor_id,
                "cam_id": cam_id,
                "arduino_idx": arduino_idx,
                "car_direction": car_direction,
                "start_line": start_line,
                "end_line": end_line,
                "is_centroid_inside": is_centroid_inside
            }

            img_restoration_result_queue.put(result)

            del plate_result
            gc.enable()
            gc.collect()
            gc.disable()            

        except Exception as e:
            print(f"Error in image_restoration: {e}")
        
    del img_restore

class ImageRestoration:
    def __init__(self):
        self.gan_model = GanModel()
        self.BASE_DIR = config.BASE_DIR
        self.DATASET_DIR = os.path.join(self.BASE_DIR, "dataset")
        self.DATASET_IMAGE_RESTORATION_DIR = os.path.join(self.DATASET_DIR, "2_image_restoration", datetime.now().strftime('%Y-%m-%d-%H'))

    def process_image(self, image, is_save=True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        color_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
        restored_image = self.gan_model.super_resolution(color_image)

        if is_save:
            self.save_restored_image(restored_image)

        return restored_image

    def save_restored_image(self, restored_image):
        if not os.path.exists(self.DATASET_IMAGE_RESTORATION_DIR):
            os.makedirs(self.DATASET_IMAGE_RESTORATION_DIR)

        if restored_image is not None and restored_image.size > 0:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            filename = os.path.join(self.DATASET_IMAGE_RESTORATION_DIR, f'{timestamp}.jpg')

            if len(restored_image.shape) == 3:
                cv2.imwrite(filename, cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR))
        else:
            logging.warning("[ImageRestoration] Restored image is empty or invalid, not saving.")
            
if __name__ == '__main__':
    # image = cv2.imread(r"D:\engine\cv\image_restoration\backup\18-48\plate_saved\2024-10-24-18-32-55-343434.jpg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # restored_image = img_restore.process_image(image)
    # cv2.imshow(restored_image)

    img_restore = ImageRestoration()
    folder_input = r"D:\engine\cv\image_restoration\backup\18-48\plate_saved"

    for f in os.listdir(folder_input):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_input, f)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            restored_image = img_restore.process_image(image)
