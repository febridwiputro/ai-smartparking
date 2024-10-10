import numpy as np
import cv2
from ultralytics import YOLO
from src.config.config import config


class PlatDetector:
    def __init__(self, plate_detection_model):
        # TODO refactor YOLO model
        self.model = plate_detection_model

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # TODO image processing
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return image_bgr

    def predict(self, image: np.ndarray):
        preprocessed_image = self.preprocess(image)
        # results = self.model.predict(preprocessed_image, conf=0.25, device="cuda:0", verbose=False, classes=config.CLASS_PLAT_NAMES)
        results = self.model.predict(preprocessed_image, conf=0.3, device="cuda:0", verbose=False)
        return results

    def draw_boxes(self, frame, results):
        for box in results.boxes.xyxy.cpu():
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # center_x = (x1 + x2) // 2
            # center_y = (y1 + y2) // 2
            # cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot

    def filter_object(self, result, selected_class_names: dict):
        if len(result.boxes) == 0:
            return result

        indices = [
            i for i, cls in enumerate(result.boxes.cls.cpu().numpy())
            if int(cls) in selected_class_names.keys()
        ]
        result.boxes = result.boxes[indices]

    def get_plat_image(self, image, results):
        for box in results.boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = map(int, box)
            plat = image[max(y1, 0): min(y2, image.shape[0]), max(x1, 0): min(x2, image.shape[1])]
            # print("plat.shape", plat.shape)
            # cv2.imshow("Plat", plat)
            return plat

        return np.array([])

