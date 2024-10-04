import uuid
from datetime import datetime

from src.config.config import config
from src.config.logger import logger
from src.controller.matrix_controller import MatrixController
from src.model.car_model import VehicleDetector
from src.model.plat_model import PlatDetector
from src.model.text_recognition_model import TextRecognition
from src.utils import *
from src.view.show_cam import show_cam, show_text, show_line
from src.Integration.service_v1.controller.floor_controller import FloorController
from src.Integration.service_v1.controller.fetch_api_controller import FetchAPIController

# semuanya berjalan secara per frame
class OCRController:
    def __init__(self, ard, matrix_total):
        self.previous_state = None
        self.current_state = None
        self.passed_a = 0
        self.plate_no = ""
        self.car_detector = VehicleDetector(config.MODEL_PATH)
        self.plat_detector = PlatDetector(config.MODEL_PATH_PLAT_v2)
        self.ocr = TextRecognition()
        self.container_plate_no = []
        # self.container_text = []
        # self.real_container_text = []
        self.mobil_masuk = False
        self.track_id = 0
        self.centroids = []
        self.passed: int = 0
        self.matrix_text = MatrixController(ard, 0, 100)
        self.matrix_text.start()
        self.matrix = matrix_total
        self.matrix.start(self.matrix.get_total())
        self.width, self.height = 0, 0
        self.status_register = False
        self.car_direction = None
        self.prev_centroid = None
        self.num_skip_centroid = 0
        self.centroid_sequence = []
        self.db_floor = FloorController()
        self.db_mysn = FetchAPIController()

    def check_floor(self, cam_idx):
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


    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(convert_bbox_to_decimal((self.height, self.width), [[[x, y]]]))
    
    def get_centroid(self, results, line_pos):
        centro = get_centroids(results, line_pos)
        estimate_tracking, self.track_id = self.car_detector.get_tracking_centroid(centro)
        # self.track_id = self.car_detector.get_id_track(results[0])
        if estimate_tracking != ():
            return centro
        else:
            return estimate_tracking
    
    def is_car_out_v2(self, boxes):
        sorted_boxes = sorted(boxes, key=lambda x: x[3], reverse=True)
        if len(sorted_boxes) > 0:
            box0 = sorted_boxes[0]
            cx, cy = (box0[0] + box0[2]) / 2, box0[3]
            d = 200
            if self.prev_centroid is not None:
                d = abs(cy - self.prev_centroid[1])
            if d < 100:
                self.prev_centroid = cx, cy
                if len(self.centroid_sequence) > 5:
                    seq = np.array(self.centroid_sequence)
                    self.centroid_sequence = []
                    dist = seq[:-1] - seq[1:]
                    negative_indices = (dist < 0).astype(int)
                    positive_indices = (dist > 0).astype(int)
                    
                    if sum(negative_indices) > sum(positive_indices):
                        # print("mobil masuk".center(100, "="))
                        return False
                    elif sum(negative_indices) < sum(positive_indices):
                        # print("mobil keluar".center(100, "#"))
                        return True
                    else:
                        # print("mobil diam".center(100, "*"))
                        return None
                
                else:
                    self.centroid_sequence.append(cy)
            else:
                self.num_skip_centroid += 1
                if self.num_skip_centroid > 5:
                    self.prev_centroid = cx, cy
                    self.num_skip_centroid = 0
                    self.centroid_sequence = []
        return None
    
    def get_car_image(self, frame, threshold=0.008):
        results = self.car_detector.predict(frame)
        if not results[0].boxes.xyxy.cpu().tolist():
            return np.array([]), results
        boxes = results[0].boxes.xyxy.cpu().tolist()
        height, width = frame.shape[:2]
        filtered_boxes = [box for box in boxes if (box[3] < height * (1 - threshold))]
        if not filtered_boxes:
            return np.array([]), results
        sorted_boxes = sorted(filtered_boxes, key=lambda x: x[3] - x[1], reverse=True)
        if len(sorted_boxes) > 0:
            box = sorted_boxes[0]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            car_frame = frame[y1:y2, x1:x2]
            if car_frame.shape[0] == 0 or car_frame.shape[1] == 0:
                return np.array([]), results
            return car_frame, results
        return np.array([]), results
    
    def get_process_plat_image(self, car, is_bitwise=True) -> (np.ndarray, np.ndarray):
        if car.shape[0] == 0 or car.shape[1] == 0:
            return np.array([])
        
        results_plat = self.plat_detector.predict(car)
        if not results_plat:
            return np.array([])
        
        plat = self.plat_detector.get_plat_image(image=car, results=results_plat[0])
        if plat.shape[0] == 0 or plat.shape[1] == 0:
            return np.array([]), np.array([])
        plat_preprocessing = self.ocr.image_processing(plat, is_bitwise)
        return plat_preprocessing, plat
    
    def image_to_text(self, frame):
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            return "", []
        
        text, results, box = self.ocr.recognition_image_text(frame)
        if not text:
            return "", box
        
        return self.ocr.filter_text(text), box
    
    def crop_frame(self, frame, cam_idx):
        # polygons_point = [ config.POINTS_BACKGROUND_LT2_OUT]

        polygons_point = [config.POINTS_BACKGROUND_LT2_IN, 
                          config.POINTS_BACKGROUND_LT2_OUT,
                          config.POINTS_BACKGROUND_LT3_IN,
                          config.POINTS_BACKGROUND_LT3_OUT,
                          config.POINTS_BACKGROUND_LT4_IN,
                          config.POINTS_BACKGROUND_LT4_OUT,
                          config.POINTS_BACKGROUND_LT5_IN,
                          config.POINTS_BACKGROUND_LT5_OUT]
        
        point=polygons_point[cam_idx]

        floor_position, cam_position = self.check_floor(cam_idx=cam_idx)
        self.height, self.width = frame.shape[:2]
        polygons = [point]
        bbox = convert_decimal_to_bbox((self.height, self.width), polygons)
        frame = crop_polygon(frame, bbox[0])

        if frame.shape[0] == () or frame.shape[1] == ():
            return "", np.array([]), np.array([])

        if floor_position == 2:
            # polygon_point = config.POLYGON_POINT_LT2_OUT
            polygon_point = config.POLYGON_POINT_LT2_IN if cam_position == "IN" else config.POLYGON_POINT_LT2_OUT
        elif floor_position == 3:
            polygon_point = config.POLYGON_POINT_LT3_IN if cam_position == "IN" else config.POLYGON_POINT_LT3_OUT
        elif floor_position == 4:
            polygon_point = config.POLYGON_POINT_LT4_IN if cam_position == "IN" else config.POLYGON_POINT_LT4_OUT
        elif floor_position == 5:
            polygon_point = config.POLYGON_POINT_LT5_IN if cam_position == "IN" else config.POLYGON_POINT_LT5_OUT
        else:
            return []

        poly_points = convert_decimal_to_bbox((self.height, self.width), polygon_point)
        
        return poly_points, frame, floor_position, cam_position


    def check_centroid_location(self, results, poly_points, inverse=False):
        point = (self.centroids[0][0], self.centroids[0][1])
        start = point_position(poly_points[0], poly_points[1], point, inverse=inverse)
        end = point_position(poly_points[2], poly_points[3], point, inverse=inverse)
        self.centroids = self.get_centroid(results, line_pos=False)
        point = (self.centroids[0][0], self.centroids[0][1])
        if inverse:
            return end, start
        else:
            return start, end

    def car_direct(self, frame, arduino_idx, cam_idx='a') -> list:
        floor_position, cam_position = self.check_floor(cam_idx=cam_idx)
        floor_id = floor_position

        floor_id = floor_position
        slot = self.db_floor.get_slot_by_id(floor_id)
        total_slot = slot["slot"]
        max_slot = slot["max_slot"]
        
        poly_points, frame, _, _ = self.crop_frame(frame, cam_idx)
        car, results = self.get_car_image(frame)

        show_text(f"Floor : {floor_position} {cam_position}", frame, 5, 50)
        show_text(f"Plate No. : {self.plate_no}", frame, 5, 100, (0, 255, 0) if self.status_register else (0, 0, 255))
        show_text(f"Parking Lot Available : {total_slot}", frame, 5, 150)
        # show_text(f"Total Car: {self.matrix.get_total()}", frame, 5, 150)
        # show_text(f"Parkir Lot Available: {available_slot}", frame, 5, 200, (0, 255, 0) if available_slot >= 1 else (0, 0, 255))

        show_line(frame, poly_points[0], poly_points[1])
        show_line(frame, poly_points[2], poly_points[3])
        show_cam(str(cam_idx), frame)
        
        # cv2.setMouseCallback(str(cam_idx), self.mouse_event)
        
        if car.shape[0] == 0 or car.shape[1] == 0:
            return []
        self.mobil_masuk = True
        cv2.imwrite(fr"D:\car\car_{time.time()}.jpg", car)
        self.centroids = self.get_centroid(results, line_pos=True)
        direction = self.is_car_out_v2(results[0].boxes.xyxy.cpu().tolist())
        if direction is not None or self.car_direction is None:
            self.car_direction = direction
        
        start, end = self.check_centroid_location(results, poly_points, inverse=self.car_direction)

        return [car, start, end]


    def processing_car_counter(self, list_data, car_direction=None):
        if car_direction is not None:
            self.car_direction = car_direction
        _, start, end = list_data

        if start and not end:
            self.passed_a = 2
        elif end:
            if self.passed_a == 2:
                self.matrix.plus_car() if not self.car_direction else self.matrix.minus_car()
            self.mobil_masuk = False
            self.passed_a = 0
        
        # logger.write(f"{self.matrix.get_total()}, {'KELUAR' if self.car_direction else 'MASUK'}, {list_data[1:-1]}".center(100, "="), logger.DEBUG)
    
    def processing_ocr(self, arduino_idx, cam_idx, frame, list_data, car_direction=None):
        if car_direction is not None:
            self.car_direction = car_direction

        car, start, end= list_data
        if car.shape[0] == 0 or car.shape[1] == 0:
            return "", frame, np.array([])
        if start and not end:
            self.passed = 2
        elif end:
            self.processing_logic_car(arduino_idx, cam_idx, self.car_direction)
            self.mobil_masuk = False

        if self.passed > 0:
            pre_plat, plat = self.get_process_plat_image(car)
            if pre_plat.shape[0] == 0 or pre_plat.shape[1] == 0:
                return "", frame, np.array([])
            text, bbox = self.image_to_text(pre_plat)
            show_cam("pre_plat", pre_plat)
            # show_polygon(pre_plat, bbox, (0, 0, 0))
            if text != "" and text is not None:
                # print("real container : ", self.real_container_text)
                # print("container plate_no : ", self.container_plate_no)
                self.container_plate_no.append(text)
            return text, frame, plat
        return "", frame, np.array([])
    
    def check_db(self, text):
        if not self.ocr.controller.check_exist_plat(license_no=text):
            closest_text = find_closest_strings_dict(text, self.ocr.all_plat)
            if len(closest_text) == 1 and list(closest_text.values())[0] <= 2:
                text = list(closest_text.keys())[0]
                return True
            else:
                return False
        else:
            # print("plat ada di DB : ", self.text)
            return True

    def send_plate_data(self, floor_id, plate_no, cam_position):
        generate_uuid = uuid.uuid4() 
        created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        params = [
            {
                "id": str(generate_uuid),
                "floor": str(floor_id),  # "1"
                "license": plate_no, # "bp1234bp"
                "zone": "1",  # "1" = kanan / "2" = kiri
                "cam": cam_position.lower(), # "in" / "out"
                "vehicle_type": "car", # "car" / "motorcycle"
                "created_date": created_date
            }
        ]
        send_date = created_date

        return self.db_mysn.send_data_to_mysn(params, send_date)


    # def processing_logic_car(self, arduino_idx, cam_idx, status_car):
    #     if not (self.passed == 2):
    #         self.passed = 0
    #         return

    #     if len(self.container_plate_no) == 0:
    #         self.passed = 0
    #         return
        
    #     self.status_register = True
    #     if len(self.container_plate_no) >= 1:
    #         plate_no = most_freq(self.container_plate_no)
    #         plate_no_detected = plate_no
    #         status_plate_no = self.check_db(plate_no_detected)
    #         if not status_plate_no:
    #             logger.write(f"Warning, plat is unregistered, reading container text !! : {plate_no}", logger.WARN)
    #             self.status_register = False

    #     # Periksa lantai saat ini dan posisi kamera (IN/OUT)
    #     floor_position, cam_position = self.check_floor(cam_idx=cam_idx)
    #     floor_id = floor_position

    #     # Mendapatkan data slot dan vehicle_total saat ini dari database
    #     slot = self.db_floor.get_slot_by_id(floor_id)
    #     total_slot = slot["slot"]
    #     max_slot = slot["max_slot"]
    #     vehicle_total = slot["vehicle_total"]  # Total kendaraan saat ini di lantai
    #     total_slot_update = total_slot

    #     print(f"Mobil dengan plat {plate_no} terdeteksi di lantai {floor_position} melalui kamera {cam_position}")

    #     # Kondisi Mobil Turun
    #     if status_car:  # Mobil turun melalui IN atau OUT
    #         # Tambahkan total_slot di lantai saat ini
    #         print(f"UPDATE +1: Mobil turun, update slot di lantai {floor_position}")
    #         total_slot_update = total_slot + 1
    #         self.db_floor.update_slot_by_id(id=floor_id, new_slot=total_slot_update)

    #         # Kurangi vehicle_total karena mobil turun
    #         new_vehicle_total = vehicle_total - 1 if vehicle_total > 0 else 0
    #         print(f"UPDATE vehicle_total -1: Total kendaraan di lantai {floor_position} = {new_vehicle_total}")
    #         self.db_floor.update_vehicle_total_by_id(floor_id, new_vehicle_total)

    #         # Lakukan penyesuaian slot di lantai di bawahnya
    #         if floor_position > 2:  # Jika bukan lantai 2
    #             prev_floor_id = floor_position - 1
    #             prev_slot = self.db_floor.get_slot_by_id(prev_floor_id)["slot"]
    #             if prev_slot > 0:  # Cek apakah ada slot yang tersedia di lantai sebelumnya
    #                 print(f"UPDATE -1: Mobil turun ke lantai {prev_floor_id}")
    #                 prev_slot_update = prev_slot - 1
    #                 self.db_floor.update_slot_by_id(id=prev_floor_id, new_slot=prev_slot_update)
    #             else:
    #                 print(f"WARNING: Lantai {prev_floor_id} sudah kosong, mobil tidak bisa turun lebih lanjut.")
        
    #     # Kondisi Mobil Naik
    #     else:  # Mobil naik melalui IN atau OUT
    #         # Kurangi total_slot di lantai saat ini
    #         if total_slot > 0:
    #             print(f"UPDATE -1: Mobil naik, update slot di lantai {floor_position}")
    #             total_slot_update = total_slot - 1
    #             self.db_floor.update_slot_by_id(id=floor_id, new_slot=total_slot_update)

    #         # Tidak perlu melakukan perubahan pada lantai di atasnya untuk kondisi mobil naik

    #     # Memperbarui total slot dan menampilkan jumlah slot yang tersedia
    #     matrix_update = MatrixController(arduino_idx, max_car=max_slot, total_car=total_slot_update)
    #     available_space = matrix_update.get_total()
    #     self.total_slot = max_slot - available_space
    #     self.plate_no = plate_no

    #     print(f"PLAT_NO : {plate_no}, AVAILABLE PARKING SPACES : {available_space}, VEHICLE_TOTAL : {vehicle_total}, STATUS : {'TAMBAH' if status_car else 'KURANG'}, FLOOR : {floor_position}, CAMERA : {cam_position}, TOTAL_FRAME: {len(self.container_plate_no)}")

    #     # Kirim data plat mobil ke sistem
    #     char = "H" if self.status_register else "M"
    #     matrix_text_text = plate_no + "," + char + ";"
    #     self.matrix_text.write_arduino(matrix_text_text)
    #     self.container_plate_no = []
    #     self.passed = 0

    #     if not self.ocr.controller.check_exist_plat(plate_no):
    #         self.status_register = False
    #         logger.write(f"WARNING THERE IS NO PLAT IN DATABASE!!! text: {plate_no}, status: {status_car}",
    #                     logger.WARNING)




    def processing_logic_car(self, arduino_idx, cam_idx, status_car):
        if not (self.passed == 2):
            self.passed = 0
            return

        if len(self.container_plate_no) == 0:
            self.passed = 0
            return
        
        self.status_register = True
        if len(self.container_plate_no) >= 1:
            plate_no = most_freq(self.container_plate_no)
            plate_no_detected = plate_no
            status_plate_no = self.check_db(plate_no_detected)
            if not status_plate_no:
                logger.write(f"Warning, plat is unregistered, reading container text !! : {plate_no}", logger.WARN)
                self.status_register = False

        # Periksa lantai saat ini dan posisi kamera (IN/OUT)
        floor_position, cam_position = self.check_floor(cam_idx=cam_idx)
        floor_id = floor_position

        # Mendapatkan data slot dan vehicle_total saat ini dari database
        slot = self.db_floor.get_slot_by_id(floor_id)
        total_slot = slot["slot"]
        max_slot = slot["max_slot"]
        vehicle_total = slot["vehicle_total"]  # Total kendaraan saat ini di lantai
        total_slot_update = total_slot
        vehicle_slot_update = vehicle_total

        print(f"Mobil dengan plat {plate_no} terdeteksi di lantai {floor_position} melalui kamera {cam_position}")
        
        # new_vehicle_total = vehicle_total - 1 if vehicle_total > 0 else 0
        # print(f"UPDATE vehicle_total -1: Total kendaraan di lantai {floor_position} = {new_vehicle_total}")
        # self.db_floor.update_vehicle_total_by_id(floor_id, new_vehicle_total)

        prev_floor_position = floor_position - 1
        prev_slot = self.db_floor.get_slot_by_id(prev_floor_position)
        prev_total_slot = slot["slot"]
        prev_max_slot = slot["max_slot"]

        next_floor_position = floor_position + 1
        next_slot = self.db_floor.get_slot_by_id(next_floor_position)
        next_total_slot = slot["slot"]
        next_max_slot = slot["max_slot"]

        # KELUAR / TURUN
        if status_car:
            if total_slot == 0:
                print("UPDATE +1")
                total_slot_update = total_slot + 1
                self.db_floor.update_slot_by_id(id=floor_position, new_slot=total_slot_update)

                # MOBIL MASUK DARI ATAS
                if prev_floor_position != 0 :
                    if prev_total_slot > 0 and prev_total_slot < prev_total_slot:
                        prev_total_slot_update = total_slot - 1
                        self.db_floor.update_slot_by_id(id=floor_position, new_slot=prev_total_slot_update)

            elif total_slot > 0 and total_slot < max_slot:
                print("UPDATE +1")
                total_slot_update = total_slot + 1
                self.db_floor.update_slot_by_id(id=floor_position, new_slot=total_slot_update)

                # MOBIL MASUK DARI ATAS
                if prev_floor_position != 0:
                    if prev_total_slot > 0 and prev_total_slot < prev_total_slot:
                        prev_total_slot_update = total_slot - 1
                        self.db_floor.update_slot_by_id(id=floor_position, new_slot=prev_total_slot_update)

            elif total_slot >= max_slot:
                print("UPDATE 0")
                total_slot_update = total_slot
                self.db_floor.update_slot_by_id(id=floor_position, new_slot=total_slot_update)

                # MOBIL MASUK DARI ATAS
                if prev_floor_position != 0:
                    if prev_total_slot > 0 and prev_total_slot < prev_total_slot:
                        prev_total_slot_update = total_slot - 1
                        self.db_floor.update_slot_by_id(id=floor_position, new_slot=prev_total_slot_update)


        matrix_update = MatrixController(arduino_idx, max_car=max_slot, total_car=total_slot_update)
        available_space = matrix_update.get_total()
        self.total_slot = max_slot - available_space
        self.plate_no = plate_no

        print(f"PLAT_NO : {plate_no}, AVAILABLE PARKING SPACES : {available_space}, STATUS : {'TAMBAH' if not status_car else 'KURANG'}, FLOOR : {floor_position}, CAMERA : {cam_position}, TOTAL_FRAME: {len(self.container_plate_no)}")
    
        self.send_plate_data(floor_id=floor_id, plate_no=plate_no, cam_position=cam_position)

        char = "H" if self.status_register else "M"
        matrix_text_text = plate_no + "," + char + ";"
        # print(matrix_text_text)
        self.matrix_text.write_arduino(matrix_text_text)
        self.container_plate_no = []
        # self.real_container_text = []
        self.passed = 0
        
        if not self.ocr.controller.check_exist_plat(plate_no):
            self.status_register = False
            logger.write(f"WARNING THERE IS NO PLAT IN DATABASE!!! text: {plate_no}, status: {status_car}",
                         logger.WARNING)




        # if total_slot < 0:
        #     total_slot_update = 0
        #     print("UPDATE 0")
        # elif total_slot == 18 and status_car:
        #     print("UPDATE 0")
        # else:
        #     if not status_car:
        #         print("UPDATE -1")
        #         total_slot_update = max(0, total_slot - 1)
        #     else:
        #         print("UPDATE +1")
        #         total_slot_update = min(18, total_slot + 1)

        # self.db_floor.update_slot_by_id(id=floor_id, new_slot=total_slot_update)


