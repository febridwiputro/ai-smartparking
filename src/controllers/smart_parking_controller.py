import threading
import time
import cv2
import numpy as np

from utils.CustomCv2 import CameraV1
from .parking_lot_controller import ParkingLot
from ..Integration.arduino import Arduino
from ..Integration.service_v1.controller.camera_controller import CameraController
from ..Integration.service_v1.controller.slot_controller import SlotController
from ..config.logger import logger
from ..model.car_model import VehicleDetector
from ..utils import convert_decimal_to_bbox, motion_detection
from ..view.show_cam import show_cam


# object smart parking percamera
class SmartParking:
    def __init__(self, video_sources: list,  parking_lots: list, slots_list: list, links_cam: [], arduino: Arduino = None):
        """

        Args:
            video_sources: list of video source
            model_path: path to model
            parking_lots: list of config bbox
            arduino: controller per area
            area: area of parking lot
            slot: list of slot from camera
        """
        self.links_cam = links_cam
        self.motion_detected = None
        print("INIT SMART PARKING OBJECT")
        self.video_sources = video_sources
        self.detector = [VehicleDetector() for _ in range(len(self.video_sources))]
        self.ard = arduino
        self.thresh_time = 10
        self.controller = SlotController()
        self.caps = [CameraV1(src, is_video=False) for src in self.video_sources]
        print("Open camera")

        # self.parking_lots_instances_list = [
        #     [ParkingLot(idx + 1, lot) for idx, lot in enumerate(parking_lot)]
        #     for parking_lot in parking_lots
        # ]

        # self.parking_lots_instances = []
        #         self.previous_states = {i['slot']: i['status'] for i in self.controller.get_status_by_area(area=self.area) if
        #                      i['slot'] in self.slot}
        #
        self.area = []
        self.name = []
        for i in self.links_cam:
            area_name = CameraController().get_area_name_by_cam_link(i)
            self.area.append(area_name[0])
            self.name.append(area_name[1])
    
        print(self.area)
        print(slots_list)
        frame = self.caps[0].get()
        if frame[0] is None or frame[1] is None:
            raise RuntimeError("Failed to retrieve frame dimensions from camera.")

        self.parking_lots_instances_list = [[ParkingLot(slots[i], lot) for i, lot in enumerate(convert_decimal_to_bbox(frame, parking_lot))] for parking_lot, slots in zip(parking_lots, slots_list)]
        # self.previous_states = [{lot.id: lot.pos for lot in parking_lots} for parking_lots in self.parking_lots_instances_list]
        self.previous_states = [{i['slot']: i['status'] for i in self.controller.get_status_by_area(area=self.area[i]) if
                                 i['slot'] in slot} for i, slot in enumerate(slots_list)]
        self.last_time = [[time.perf_counter() for _ in range(len(self.video_sources))] for _ in self.parking_lots_instances_list]
        
        # for i in self.links_cam:
        # #     area, name = CameraController().get_area_name_by_cam_link(i)
        # for links, prev in zip(self.links_cam ,self.previous_states):
        #     area, name = CameraController().get_area_name_by_cam_link(links)
        #     for j, status in prev.items():
        #         a = f"S,{j},{area};" if status else f"M,{j},{area};"
        #         self.ard.write(a)
        #         logger.write(a, logger.DEBUG)
                
        
                


        self.threads = []
        for idx, cap in enumerate(self.caps):
            cap.start()
            thread = threading.Thread(target=self.process_video, args=(cap, idx, self.detector[idx]))
            self.threads.append(thread)
            thread.start()



        # for i, value in self.previous_states.items():
        #     self.ard.write(f"M,{i}" if not value else f"S,{i}")

        # Cannot debug if using cam from mr Maftuh
        # for idx, cap in enumerate(self.caps):
        #     cap.start()
        #     if not cap.isOpened():
        #         logger.write(f"Error: Could not open video source {self.video_sources[idx]}", logger.WARNING)
        #         quit()
        # Start a thread for each video capture


    def run(self):
        for thread in self.threads:
            thread.join()

        for cap in self.caps:
            cap.release()
        cv2.destroyAllWindows()


    def process_video(self, cap, idx, detector):
        print("PROCESS VIDEO ", end=", TOTAL THREAD :")
        initial_state_sent = False
        print(threading.active_count())
        controller = SlotController()
        cam_controller = CameraController()
        area, name = cam_controller.get_area_name_by_cam_link(self.links_cam[idx])
        last_tim = time.perf_counter()
        last_time_check = time.perf_counter()
        avg = None
        parking_lots_instances = self.parking_lots_instances_list[idx]
        previous_states = self.previous_states[idx]
        while True:
            ret, frame = cap.read()
            # sending arduino message
            if self.ard.texts is not []:
                if time.perf_counter() - last_tim <= 1:
                    self.ard.sending()
                    last_tim = time.perf_counter()
                    
            # processing frame to get status
            if frame is not None:
                frame_with_boxes = frame.copy()
                avg, self.motion_detected = motion_detection(frame, avg)
                if not initial_state_sent:
                    for key, value in previous_states.items():
                        a = f"S,{key},{area};" if value else f"M,{key},{area};"
                        self.ard.write(a)
                    initial_state_sent = True
                
                if self.motion_detected:
                    result = detector.tracking(frame)
                    frame_with_boxes = detector.draw_boxes(frame, result[0])

                    for index, lot in enumerate(parking_lots_instances):
                        frame_with_boxes = lot.update_occupancy(frame_with_boxes, result[0])
                        if previous_states[lot.id] != lot.pos:
                            if self.last_time[idx][index] == 0:
                                self.last_time[idx][index] = time.perf_counter()
                            elapsed_time = time.perf_counter() - self.last_time[idx][index]
                            if elapsed_time > self.thresh_time:
                                print(f"Number {lot.id} : Status {previous_states[lot.id]}")
                                controller.update_slot_status(area=area, slot=int(lot.id), status=lot.pos,
                                                              updateby="update_by_controller")
                                self.ard.write(f"S,{lot.id},{area};" if lot.pos else f"M,{lot.id},{area};")
                                previous_states[lot.id] = lot.pos
                                self.last_time[idx][index] = 0  # Reset the timer after action completes
                    
                # checking without motion detection per cams/thread
                if time.perf_counter() - last_time_check >= 120:
                    for lot in parking_lots_instances:
                        if previous_states[lot.id] != lot.pos:
                            print(f"Number {lot.id} : Status {previous_states[lot.id]}")
                            controller.update_slot_status(area=area, slot=int(lot.id), status=lot.pos,
                                                          updateby="update_by_controller")
                            self.ard.write(f"S,{lot.id},{area};" if lot.pos else f"M,{lot.id},{area};")
                            previous_states[lot.id] = lot.pos
                    last_time_check = time.perf_counter()
                    
                
                
                for lot in parking_lots_instances:
                    color = (0, 255, 0) if not lot.occupied else (0,0,255)
                    cv2.polylines(frame, [lot.polygon.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=2)
                    text_pos =  tuple(lot.polygon[0])
                    cv2.putText(frame, str(lot.id), (text_pos[0], text_pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                window_name = f"Smart Parking - {name} - {area}"
                frame_with_boxes = cv2.resize(frame_with_boxes, (640, 360))
                show_cam(window_name, frame_with_boxes)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        cap.release()




# import cv2
# import time
# import os
# import sys
#
# from ultralytics import YOLO
#
# from utils.CustomCv2 import CameraV1
# from .parking_lot_controller import ParkingLot
# from ..Integration.arduino import Arduino
# from ..Integration.service_v1.controller.slot_controller import SlotController
# from ..config.config import config
# from ..utils import convert_decimal_to_bbox, motion_detection
#
# sys.path.append(os.path.abspath(os.curdir))
#
# # S,1,LANTAI_1;S,2,LANTAI_1;S,2,LANTAI_1;
#
# class SmartParking:
#     def __init__(self, video_source, model_path, parking_lots, area, slot, arduino_object: Arduino):
#         """
#         Args:
#             video_source: source cam
#             model_path: model path for YOLO
#             parking_lots: bbox per slot
#             area: 1 area (lantai)
#             slot: all Slot per area
#             arduino_object : Arduino object per area
#         """
#         self.thresh_time = 1
#         self.avg = None
#         self.motion_detected = False
#         self.slot = slot
#         self.last_time = time.time()
#         # self.ard = arduino_object
#         self.controller = SlotController()
#         self.area = area
#         self.cam_prefix = config.LINK_CAM_PREFIX
#         self.video_source = f"{self.cam_prefix}{video_source}"
#         self.model = YOLO(model_path, task="detect")
#         self.cap = CameraV1(self.video_source)
#         self.cap.start()
#
#         if self.cap.isOpened:
#             ret, self.frame = self.cap.read()
#         else:
#             raise ValueError("Video source not found")
#
#         self.parking_lots_instances = [ParkingLot(slot[i], lot) for i, lot in enumerate(convert_decimal_to_bbox(self.frame, parking_lots))]
#         self.previous_states = {i['slot']: i['status'] for i in self.controller.get_status_by_area(area=self.area) if
#                      i['slot'] in self.slot}
#
#         for i, value in self.previous_states.items():
#             # self.ard.write(f"M,{i}" if not value else f"S,{i}")
#             print(f"M,{i}" if not value else f"S,{i}")
#
#
#
#     @staticmethod
#     def filter_object(result, selected_class_names):
#         if len(result.boxes) == 0:
#             return result
#
#         indices = [
#             i for i, cls in enumerate(result.boxes.cls.cpu().numpy())
#             if int(cls) in selected_class_names.keys()
#         ]
#         result.boxes = result.boxes[indices]
#         return result
#
#
#     def detection(self, frame, initial_state_sent=False):
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
#         results = self.model.predict(frame_bgr, verbose=False, task='detection', conf=0.3,iou=0.5, classes= [0])
#
#         result = self.filter_object(results[0], config.CLASS_NAMES)
#         for box in results[0].boxes.data.tolist():
#             x1, y1, x2, y2, scr, cls = box
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), thickness=5, color=(255, 0, 0))
#
#
#         for lot in self.parking_lots_instances:
#             frame = lot.update_occupancy(frame, result)
#
#         if not initial_state_sent:
#             for lot in self.parking_lots_instances:
#                 self.previous_states[lot.id] = lot.pos
#
#
#         for lot in self.parking_lots_instances:
#             if self.previous_states[lot.id] != lot.pos:
#                 if time.time() - self.last_time > self.thresh_time:
#                     print(f"Number {lot.id} : Status {self.previous_states[lot.id]}")
#                     self.controller.update_slot_status(area=self.area, slot=int(lot.id), status = lot.pos, updateby="test_update_by_controller")
#                     # self.ard.write(f"S,{lot.id}" if lot.pos else f"M,{lot.id}")
#                     self.previous_states[lot.id] = lot.pos
#                     # print(self.ard.read())
#                     self.last_time = time.time()
#
#         return frame
#
#
#     def run(self):
#         initial_state_sent = False
#         name = f"{self.video_source.strip(self.cam_prefix)} | {self.area}"
#         while self.cap.isOpened:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
#
#             self.avg , self.motion_detected = motion_detection(frame, self.avg)
#             if self.motion_detected:
#                 frame = self.detection(frame, initial_state_sent)
#
#                 if not initial_state_sent:
#                     initial_state_sent = True
#             cv2.imshow(name, cv2.resize(frame, (720, 500)))
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 cv2.destroyWindow(name)
#                 break
#             if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
#                 break
#
#         self.cap.release()
#
