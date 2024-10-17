import numpy as np
import cv2
from ultralytics import YOLO
from src.config.config import config
from norfair import Detection, Tracker
from utils.centroid_tracking import CentroidTracker


class VehicleDetector:
    def __init__(self, yolo_model):
        self.model = yolo_model
        self.tracking_norfair = Tracker(distance_function='euclidean', distance_threshold=10)
        self.centroid_tracking = CentroidTracker(maxDisappeared=75)
        self.detection = []
    
    def _get_tracking_centroid(self, centroids):
        self.detection.clear()
        for centroid in centroids:
            self.detection.append(Detection(np.array(centroid)))
        tracking_object = self.tracking_norfair.update(self.detection)
        coordinates = [i.estimate for i in tracking_object]
        id = [i.id for i in tracking_object]
        extracted_coordinates = [list(item.flatten()) for item in coordinates]
        return extracted_coordinates, id
    
    def get_tracking_centroid(self, centroids):
        centroids = np.array(centroids)
        track_object = [self.centroid_tracking.update(centroid.reshape(1, -1)) for centroid in centroids][0]
        id = [i for i in track_object.keys()]
        point = [list(i.flatten()) for i in track_object.values()]
        # print("point: ", point, "id :", id)
        return point, id
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return image_bgr

    def predict(self, image: np.ndarray):
        preprocessed_image = self.preprocess(image)
        results = self.model.predict(preprocessed_image, conf=0.25, device="cuda:0", verbose=False, classes=config.CLASS_NAMES)

        for result in results:
            # boxes = result.boxes  # xyxy format (x_min, y_min, x_max, y_max)
            # confidences = result.boxes.conf 
            # class_ids = result.boxes.cls 
            # for i, box in enumerate(boxes):
            #     confidence = confidences[i].item() 
            #     class_id = int(class_ids[i].item())

            #     if class_id < len(config.CLASS_NAMES):
            #         class_name = config.CLASS_NAMES[class_id] 
            #         print(f"Detected object: {class_name}, Confidence: {confidence:.2f}")
            #     else:
            #         print(f"Warning: Detected object with invalid class_id {class_id}, Confidence: {confidence:.2f}")

                self.draw_boxes(frame=image, results=result)

        return results

    # def predict(self, image: np.ndarray):
    #     preprocessed_image = self.preprocess(image)
    #     results = self.model.predict(preprocessed_image, conf=0.25, device="cuda:0", verbose = False, classes=config.CLASS_NAMES)
    #     for result in results:
    #         self.draw_boxes(frame=image, results=result)
        
    #     # self.get_vehicle_image(image=image, results=results)
    #     return results

    def tracking(self, image: np.ndarray):
        preprocess_image = self.preprocess(image)
        results = self.model.track(preprocess_image,
                                   conf=0.25,
                                   device="cuda:0",
                                   verbose = False,
                                   classes=config.CLASS_NAMES,
                                   persist=True,
                                   tracker="bytetrack.yaml"
                                   )
        return results

    def draw_all_boxes(self, frame, results):
        for box in results.boxes.xyxy.cpu():
            x1, y1, x2, y2 = map(int, box)
            color = (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    def draw_boxes(self, frame, results):
        #     # center_x = (x1 + x2) // 2
        #     # center_y = (y1 + y2) // 2
        #     # cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot

        # return frame
        # box = results[0].boxes.xyxy.cpu().tolist()
        # x1, y1, x2, y2 = map(int, box[0])
        # color = (0, 255, 0)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)\
        
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy())
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            # label = CLASS_NAMES[cls_id]
            color = (255, 255, 255)  # Green color for bounding box
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot
        
        return frame

    def get_vehicle_image(self, image,  results):
        box = results[0].boxes.xyxy.cpu().tolist()
        x1, y1, x2, y2 = map(int, box[0])
        car = image[max(y1, 0): min(y2, image.shape[0]), max(x1, 0): min(x2, image.shape[1])]
        # print("car.shape", car.shape)
        # cv2.imshow("Car", car)
        color = (255, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)

        return car


    def get_id_track(self, results):
        id = results[0].boxes.id
        if id is not None:
            return id.int().cpu().tolist()
    
    def calculate_distance_object(self, image, centroids):
        """
        Calculate distances based on the height (y-coordinate) only.

        :param image: The image on which to draw lines.
        :param centroids: List of detected centroids in the format [(x_center, y_center), ...].
        :return: List of distances and their corresponding indices, sorted by distance.
        """
        height, width = image.shape[:2]
        half_width = width / 3
        distances = []
        point1 = np.array([half_width, height - 20])
        
        for i, centroid in enumerate(centroids):
            point2 = np.array([half_width, centroid[1]])
            dist = np.linalg.norm(point1 - point2)
            cv2.line(image, pt1=(int(half_width), int(height)), pt2=(int(half_width), int(centroid[1])),
                     lineType=cv2.LINE_AA, thickness=2, color=(255, 255, 255))
            # print(dist)
            if dist < 100:
                pass
            else:
                distances.append((i, dist))  # Store index and distance
        
        # Sort distances by the distance value
        distances.sort(key=lambda x: x[1])
        return distances

       

    def get_boxes(self, results):
        box = results[0].boxes.xyxy.cpu().tolist()
        return box[0]