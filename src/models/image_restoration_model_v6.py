import os
import cv2
import numpy as np
import logging
from datetime import datetime
import gc

from src.config.config import config
from src.config.logger import logger
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
            plate_image = plate_result.get("frame")
            bg_color = plate_result.get("bg_color")
            floor_id = plate_result.get("floor_id")
            cam_id = plate_result.get("cam_id")
            arduino_idx = plate_result.get("arduino_idx")
            car_direction = plate_result.get("car_direction")
            start_line = plate_result.get("start_line")
            end_line = plate_result.get("end_line")

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
        self.saved_dir = 'image_restoration_saved'

    def process_image(self, image, is_save=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

        restored_image = self.gan_model.super_resolution(resized_image)

        if is_save:
            self.save_restored_image(restored_image)

        return restored_image

    def save_restored_image(self, restored_image):
        if not os.path.exists(self.saved_dir):
            os.makedirs(self.saved_dir)

        if restored_image is not None and restored_image.size > 0:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            filename = os.path.join(self.saved_dir, f'{timestamp}.jpg')

            if len(restored_image.shape) == 2:  # Grayscale
                cv2.imwrite(filename, restored_image)
            elif len(restored_image.shape) == 3:  # Color
                cv2.imwrite(filename, cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR))

            # logging.info(f"[ImageRestoration] Image saved as {filename}")
        else:
            logging.warning("[ImageRestoration] Restored image is empty or invalid, not saving.")





# def image_restoration(stopped, vehicle_plate_result_queue, img_restoration_result_queue):
#     img_restore = ImageRestoration()
#     frame_count = 0
#     prev_object_id = None

#     while not stopped.is_set():
#         try:
#             vehicle_plate_data = vehicle_plate_result_queue.get()

#             if vehicle_plate_data is None:
#                 continue

#             object_id = vehicle_plate_data.get('object_id')
#             bg_color = vehicle_plate_data.get('bg_color')
#             vehicle_plate_frame = vehicle_plate_data.get('frame')
#             floor_id = vehicle_plate_data.get('floor_id', 0)
#             cam_id = vehicle_plate_data.get('cam_id', "")
#             arduino_idx = vehicle_plate_data.get('arduino_idx')
#             car_direction = vehicle_plate_data.get('car_direction')
#             start_line = vehicle_plate_data.get('start_line', False)
#             end_line = vehicle_plate_data.get('end_line', False)

#             print(f"cur object_id: {object_id}, prev object_id: {prev_object_id}")

#             if object_id != prev_object_id:
#                 frame_count = 0
#                 prev_object_id = object_id

#             if vehicle_plate_frame is not None:
#                 try:
#                     restored_image = img_restore.process_image(vehicle_plate_frame)

#                     if restored_image is not None and restored_image.size > 0:
#                         if frame_count < 10:
#                             result = {
#                                 "object_id": object_id,
#                                 "bg_color": bg_color,
#                                 "frame": restored_image,
#                                 "floor_id": floor_id,
#                                 "cam_id": cam_id,
#                                 "arduino_idx": arduino_idx,
#                                 "car_direction": car_direction,
#                                 "start_line": start_line,
#                                 "end_line": end_line
#                             }

#                             img_restoration_result_queue.put(result)
#                             frame_count += 1
#                             print(f"Saved and put data for object_id: {object_id}, frame_count: {frame_count}")
#                         else:
#                             print(f"Skipping saving for object_id: {object_id}, frame_count: {frame_count}")
#                             break
#                     else:
#                         print(f"Restored image is None or empty for object_id: {object_id}")
#                 except Exception as e:
#                     print(f"Error restoring image for object_id: {object_id}: {str(e)}")
#             else:
#                 print(f"No valid vehicle plate frame for object_id: {object_id}")
#         except Exception as e:
#             print(f"Error in plate detection: {e}")