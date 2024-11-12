import os
import cv2
from src.Integration.arduino import Arduino
from src.config.config import config
from src.controllers.matrix_controller import MatrixController
from src.controllers.detection_controller import DetectionController
from src.controllers.utils.util import check_background, check_floor
from src.models.cam_model import CameraV1
from src.controllers.utils.util import check_floor
from src.view.show_cam import show_cam, show_text, show_line
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.fetch_api_controller import FetchAPIController
from src.Integration.service_v1.controller.vehicle_history_controller import VehicleHistoryController



def main():
    IS_DEBUG = True

    db_floor = FloorController()
    db_mysn = FetchAPIController()
    db_vehicle_history = VehicleHistoryController()

    if IS_DEBUG:
        video_source = config.VIDEO_SOURCE_PC
        # video_source = config.VIDEO_SOURCE_LAPTOP
        # video_source = config.VIDEO_SOURCE_20241004
    else:
        video_source = config.CAM_SOURCE_LT

    # Initialize camera captures
    caps = [CameraV1(video, is_video=True) for video in video_source]

    # Start the cameras
    for cap in caps:
        print("Starting camera: ", cap)
        cap.start()

    plat_detects = [None for _ in range(len(video_source))]
    arduino = [Arduino(baudrate=115200, serial_number=serial) for serial in config.SERIALS]
    frames = [None for _ in range(len(caps))]
    total_slots = {}

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

        matrix_controller = MatrixController(arduino_matrix=arduino_matrix, max_car=18, total_car=total_slot)
        plat_detects[i] = DetectionController(arduino_text, matrix_total=matrix_controller)
        plat_detects[i].start()

    try:
        print("Going To Open Cam")
        while True:
            for i in range(len(caps)):
                idx, cam_position = check_floor(i)
                ret, frames[i] = caps[i].read()

                if not ret:
                    print(f"Failed to read frame from camera {i}")
                    continue

                # frames[i] = cv2.resize(frames[i], (1080, 720))
                if plat_detects[i] is not None:
                    plat_detects[i].process_frame(frames[i], floor_id=idx, cam_id=cam_position)

                    results = plat_detects[i].get_results()
                    # if results is not None:
                    #     print(f"FLOOR {idx} {cam_position} : {results}")

                # Check if 'q' key is pressed to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit key 'q' pressed. Stopping...")
                    raise KeyboardInterrupt  # Trigger shutdown

    except KeyboardInterrupt:
        print("Terminating...")

    finally:
        # Release resources
        for plat_detect in plat_detects:
            if plat_detect:
                plat_detect.stop()

        for cap in caps:
            cap.release()

        cv2.destroyAllWindows()
        print("All resources released and program terminated.")


if __name__ == "__main__":
    main()