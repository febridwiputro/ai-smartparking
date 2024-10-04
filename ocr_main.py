import argparse
import threading
import time
import cv2
from src.Integration.arduino import Arduino
from src.config.config import config
from src.controller.matrix_controller import MatrixController
from src.view.show_cam import show_cam
from src.controller.ocr_controller import OCRController
from src.model.cam_model import CameraV1
from src.model.matrix_model import CarCounterMatrix
from src.Integration.service_v1.controller.floor_controller import FloorController


def main():
    db_floor = FloorController()
    IS_DEBUG = True

    if IS_DEBUG:
        caps = [CameraV1(video, is_video=True) for video in config.VIDEO_SOURCE_23]
    else:
        caps = [CameraV1(video, is_video=False) for video in config.cam_source]

    for cap in caps:
        print("cap: ", cap)

        print(f"camera {cap} : thread total: {threading.active_count()}")
        cap.start()

    arduino = [Arduino(baudrate=115200, serial_number=serial) for serial in config.serials]

    plat_detects = [None for _ in range(len(config.video_source))]
    frames = [None for _ in range(len(caps))]

    total_slot = None
    total_slots = {}

    try:
        print("Going To Open Cam")
        while True:
            for i in range(len(caps)):
                if i < 2:  # Cameras 0 and 1
                    idx = 2
                elif i < 4:  # Cameras 2 and 3
                    idx = 3
                elif i < 6:  # Cameras 4 and 5
                    idx = 4
                elif i < 8:  # Cameras 6 and 7
                    idx = 5
                else:
                    print(f"No Arduino available for camera index {i}.")
                    continue

                # Get the corresponding Arduino for the camera
                arduino_index = (i // 2) * 2  # Adjust to use the correct Arduino
                arduino_text = arduino[arduino_index]
                arduino_matrix = arduino[arduino_index + 1]

                if idx not in total_slots:
                    slot = db_floor.get_slot_by_id(idx)
                    print(f'idx {idx}, slot {slot}')
                    total_slot = slot["slot"] 
                    total_slots[idx] = total_slot

                if plat_detects[i] is None:
                    matrix_controller = MatrixController(arduino_matrix=arduino_matrix, max_car=18, total_car=total_slot)
                    plat_detects[i] = OCRController(arduino_text, matrix_total=matrix_controller)

                    print(f"Successfully sent total_slot {total_slot} to matrix_controller for camera {i}.")
 
                _, frames[i] = caps[i].read()
                # frames[i] = cv2.resize(frames[i], (1080, 720))
                if frames[i] is not None:
                    vehicle_detected = plat_detects[i].car_direct(frames[i], arduino_idx=arduino_text, cam_idx=i)

                    if vehicle_detected:
                        # print("vehicle_detected: ", vehicle_detected[1], vehicle_detected[2])
                        plat_detects[i].processing_car_counter(vehicle_detected)
                        plat_detects[i].processing_ocr(arduino_text, i, frames[i], vehicle_detected)
                        # plate_no = plat_detects[i].text
        
                        # if plate_no != prev_plate_no:
                        #     if i == 0:
                        #         if i != prev_idx:
                        #             print("PLATE NO===== ", plate_no)

                        #             slot = db_floor.get_slot_by_id(idx)
                        #             total_slot = slot["slot"]
                        #             # print(f"Updated total_slot for idx={idx}: {total_slot}")

                        #             matrix_controller = MatrixController(arduino_matrix=arduino_matrix, max_car=18, total_car=total_slot)
                        #             plat_detects[i].matrix = matrix_controller           
                        #             # total_slot = matrix_controller.get_total()

                        #             prev_plate_no = plate_no
                        #             prev_idx = i
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 32:
                cv2.waitKey(0)
        
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
    
    except KeyboardInterrupt:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        print("KeyboardInterrupt")
        exit(0)


if __name__ == "__main__":
    main()