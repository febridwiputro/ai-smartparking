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


def check_floor(cam_idx):
    if cam_idx == 0:
        return 2, "IN"
    elif cam_idx == 1:
        return 2, "OUT"
    elif cam_idx == 2:
        return 3, "IN"
    elif cam_idx == 3:
        return 3, "OUT"
    elif cam_idx == 4:
        return 4, "IN"
    elif cam_idx == 5:
        return 4, "OUT"
    elif cam_idx == 6:
        return 5, "IN"
    elif cam_idx == 7:
        return 5, "OUT"
    else:
        return 0, ""


def main():
    db_floor = FloorController()
    IS_DEBUG = True

    if IS_DEBUG:
        video_source = config.VIDEO_SOURCE_20241004
        # video_source = config.VIDEO_SOURCE_LAPTOP
        caps = [CameraV1(video, is_video=True) for video in video_source]
    else:
        video_source = config.CAM_SOURCE
        caps = [CameraV1(video, is_video=False) for video in video_source]

    plat_detects = [None for _ in range(len(video_source))]

    # Start all camera streams first
    for cap in caps:
        print("Starting camera: ", cap)
        cap.start()

    arduino = [Arduino(baudrate=115200, serial_number=serial) for serial in config.SERIALS]
    frames = [None for _ in range(len(caps))]
    total_slots = {}

    # try:
    #     print("Processing frames from all cameras")
    #     while True:
    #         for i in range(len(caps)):
    #             # Wait until all cameras are opened
    #             if not caps[i].isOpened:  # Changed from is_opened() to isOpened
    #                 print(f"Camera {i} is not opened yet.")
    #                 continue

    #             idx, cam_position = check_floor(i)

    #             if idx not in total_slots:
    #                 slot = db_floor.get_slot_by_id(idx)
    #                 print(f'idx {idx}, slot {slot}')
    #                 total_slot = slot["slot"]
    #                 total_slots[idx] = total_slot

    #             arduino_index = idx
    #             arduino_text = arduino[arduino_index]
    #             arduino_matrix = arduino[arduino_index + 1]

    #             if plat_detects[i] is None:
    #                 # Initialize MatrixController and OCRController
    #                 matrix_controller = MatrixController(arduino_matrix=arduino_matrix, max_car=18, total_car=total_slot)
    #                 plat_detects[i] = OCRController(arduino_text, matrix_total=matrix_controller)
    #                 print(f"Successfully sent total_slot {total_slot} to matrix_controller for camera {i}.")

    #             _, frames[i] = caps[i].read()
                
    #             if frames[i] is not None:
    #                 vehicle_detected = plat_detects[i].car_direct(frames[i], arduino_idx=arduino_text, cam_idx=i)

    #                 if vehicle_detected:
    #                     plat_detects[i].processing_car_counter(vehicle_detected)
    #                     plat_detects[i].processing_ocr(arduino_text, i, frames[i], vehicle_detected)

    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord("q"):
    #             break
    #         elif key == 32:
    #             cv2.waitKey(0)

    #     for cap in caps:
    #         cap.release()
    #     cv2.destroyAllWindows()

    # except KeyboardInterrupt:
    #     for cap in caps:
    #         cap.release()
    #     cv2.destroyAllWindows()
    #     print("KeyboardInterrupt")
    #     exit(0)

    try:
        print("Going To Open Cam")
        while True:
            for i in range(len(caps)):

                idx, cam_position = check_floor(i)

                if idx not in total_slots:
                    slot = db_floor.get_slot_by_id(idx)
                    print(f'idx {idx}, slot {slot}')
                    total_slot = slot["slot"]
                    total_slots[idx] = total_slot

                arduino_index = idx
                arduino_text = arduino[arduino_index]
                arduino_matrix = arduino[arduino_index + 1]
                if plat_detects[i] is None:

                    # Initialize MatrixController and OCRController
                    matrix_controller = MatrixController(arduino_matrix=arduino_matrix, max_car=18, total_car=total_slot)
                    plat_detects[i] = OCRController(arduino_text, matrix_total=matrix_controller)
                    print(f"Successfully sent total_slot {total_slot} to matrix_controller for camera {i}.")

                _, frames[i] = caps[i].read()
                # frames[i] = cv2.resize(frames[i], (1080, 720))                
                if frames[i] is not None:
                    vehicle_detected = plat_detects[i].car_direct(frames[i], arduino_idx=arduino_text, cam_idx=i)

                    if vehicle_detected:
                        plat_detects[i].processing_car_counter(vehicle_detected)
                        plat_detects[i].processing_ocr(arduino_text, i, frames[i], vehicle_detected)

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
