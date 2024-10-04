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
from src.view.show_cam import show_cam, show_text, show_line


def main():
    db_floor = FloorController()
    IS_DEBUG = True

    if IS_DEBUG:
        caps = [CameraV1(video, is_video=True) for video in config.video_source2]
    else:
        caps = [CameraV1(video, is_video=False) for video in config.cam_source]

    for cap in caps:
        cap.start()

    arduino = [Arduino(baudrate=115200, serial_number=serial) for serial in config.serials]

    plat_detects = [None for _ in range(len(config.video_source))]
    frames = [None for _ in range(len(caps))]

    total_slots = {}

    # Variabel untuk menyimpan nilai terakhir dari plate_no dan total_slot
    last_plate_no = None
    last_total_slot = None

    try:
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
                    continue

                arduino_index = (i // 2) * 2
                arduino_text = arduino[arduino_index]
                arduino_matrix = arduino[arduino_index + 1]

                if idx not in total_slots:
                    slot = db_floor.get_slot_by_id(idx)
                    total_slot = slot["slot"]
                    total_slots[idx] = total_slot

                if plat_detects[i] is None:
                    matrix_controller = MatrixController(arduino_matrix=arduino_matrix, max_car=18, total_car=total_slot)
                    plat_detects[i] = OCRController(arduino_text, matrix_total=matrix_controller)

                _, frames[i] = caps[i].read()

                if frames[i] is not None:
                    # Memanggil car_direct dan mendapatkan informasi
                    frame, plate_no, total_slot, floor_position, cam_position, poly_points, bbox = plat_detects[i].car_direct(frames[i], arduino_idx=arduino_text, cam_idx=i)

                    if plate_no:
                        last_plate_no = plate_no
                    if total_slot is not None:
                        last_total_slot = total_slot

                    if frame is None:
                        continue

                    # Menampilkan informasi pada frame
                    show_text(f"Floor: {floor_position} {cam_position}", frame, 5, 50)
                    # if last_plate_no:
                    status_register = plat_detects[i].status_register
                    show_text(f"Plate No.: {last_plate_no}", frame, 5, 100, (0, 255, 0) if status_register else (0, 0, 255))
                    # else:
                    #     show_text("Plate No.: N/A", frame, 5, 100, (0, 0, 255))

                    show_text(f"Total Car: {last_total_slot if last_total_slot is not None else 'N/A'}", frame, 5, 150)

                    # Menampilkan garis pada frame hanya jika poly_points tidak kosong
                    if poly_points is not None and len(poly_points) > 0:
                        show_line(frame, poly_points[0], poly_points[1])
                        show_line(frame, poly_points[2], poly_points[3])

                    show_cam(str(i), frame)

                    # Pastikan car adalah gambar kendaraan
                    car, results = plat_detects[i].get_car_image(frames[i])

                    if car.shape[0] > 0 and car.shape[1] > 0:
                        vehicle_detected = [last_plate_no, last_total_slot]  # Mengirim dua nilai
                        plat_detects[i].processing_car_counter(vehicle_detected)
                        plat_detects[i].processing_ocr(arduino_text, i, car, vehicle_detected)  # Mengirim car sebagai array





# def main():
#     db_floor = FloorController()
#     IS_DEBUG = True

#     if IS_DEBUG:
#         caps = [CameraV1(video, is_video=True) for video in config.video_source2]
#     else:
#         caps = [CameraV1(video, is_video=False) for video in config.cam_source]

#     for cap in caps:
#         print("cap: ", cap)

#         print(f"camera {cap} : thread total: {threading.active_count()}")
#         cap.start()

#     arduino = [Arduino(baudrate=115200, serial_number=serial) for serial in config.serials]

#     plat_detects = [None for _ in range(len(config.video_source))]
#     frames = [None for _ in range(len(caps))]


#     previous_i = None 
#     prev_plate_no = None
#     prev_idx = 0
#     total_slot = None
#     initialized = False
#     total_slots = {}

#     try:
#         print("Going To Open Cam")
#         while True:
#             for i in range(len(caps)):
#                 if i < 2:  # Cameras 0 and 1
#                     idx = 2
#                 elif i < 4:  # Cameras 2 and 3
#                     idx = 3
#                 elif i < 6:  # Cameras 4 and 5
#                     idx = 4
#                 elif i < 8:  # Cameras 6 and 7
#                     idx = 5
#                 else:
#                     print(f"No Arduino available for camera index {i}.")
#                     continue

#                 # Get the corresponding Arduino for the camera
#                 arduino_index = (i // 2) * 2  # Adjust to use the correct Arduino
#                 arduino_text = arduino[arduino_index]
#                 arduino_matrix = arduino[arduino_index + 1]

#                 if idx not in total_slots:
#                     slot = db_floor.get_slot_by_id(idx)
#                     print(f'idx {idx}, slot {slot}')
#                     total_slot = slot["slot"] 
#                     total_slots[idx] = total_slot

#                 if plat_detects[i] is None:
#                     matrix_controller = MatrixController(arduino_matrix=arduino_matrix, max_car=18, total_car=total_slot)
#                     plat_detects[i] = OCRController(arduino_text, matrix_total=matrix_controller)

#                     print(f"Successfully sent total_slot {total_slot} to matrix_controller for camera {i}.")
 
#                 _, frames[i] = caps[i].read()
#                 # frames[i] = cv2.resize(frames[i], (1080, 720))
#                 if frames[i] is not None:
#                     vehicle_detected = plat_detects[i].car_direct(frames[i], arduino_idx=arduino_text, cam_idx=i)

#                     if vehicle_detected:
#                         # print("vehicle_detected: ", vehicle_detected[1], vehicle_detected[2])
#                         plat_detects[i].processing_car_counter(vehicle_detected)
#                         plat_detects[i].processing_ocr(arduino_text, i, frames[i], vehicle_detected)
#                         # plate_no = plat_detects[i].text
        
#                         # if plate_no != prev_plate_no:
#                         #     if i == 0:
#                         #         if i != prev_idx:
#                         #             print("PLATE NO===== ", plate_no)

#                         #             slot = db_floor.get_slot_by_id(idx)
#                         #             total_slot = slot["slot"]
#                         #             # print(f"Updated total_slot for idx={idx}: {total_slot}")

#                         #             matrix_controller = MatrixController(arduino_matrix=arduino_matrix, max_car=18, total_car=total_slot)
#                         #             plat_detects[i].matrix = matrix_controller           
#                         #             # total_slot = matrix_controller.get_total()

#                         #             prev_plate_no = plate_no
#                         #             prev_idx = i
            
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