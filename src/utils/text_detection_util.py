import os
import cv2
from datetime import datetime

from src.config.logger import logger


def filter_height_bbox(bounding_boxes, verbose=False):
    converted_bboxes = []
    w_h = []
    for bbox_group in bounding_boxes:
        for bbox in bbox_group:
            if len(bbox) == 4:
                x_min, x_max, y_min, y_max = bbox
                top_left = [x_min, y_min]
                top_right = [x_max, y_min]
                bottom_right = [x_max, y_max]
                bottom_left = [x_min, y_max]
                converted_bboxes.append([top_left, top_right, bottom_right, bottom_left])

                width_bbox = x_max - x_min
                height_bbox = y_max - y_min 

                if height_bbox >= 10:
                    w_h.append(height_bbox)

def filter_readtext_frame(texts: list, verbose=False) -> list:
    w_h = []
    sorted_heights = []
    avg_height = ""
    for t in texts:
        (top_left, top_right, bottom_right, bottom_left) = t[0]
        top_left = tuple([int(val) for val in top_left])
        bottom_left = tuple([int(val) for val in bottom_left])
        top_right = tuple([int(val) for val in top_right])
        height_f = bottom_left[1] - top_left[1]
        width_f = top_right[0] - top_left[0]

        if height_f >= 10:
            w_h.append(height_f)

    if len(w_h) == 1:
        list_of_height = w_h
        filtered_heights = w_h
        sorted_heights = w_h

    elif len(w_h) == 2:
        h1, h2 = w_h
        if abs(h1 - h2) <= 10:
            filtered_heights = [h1, h2]
        else:
            filtered_heights = [max(w_h)]

        list_of_height = w_h
        sorted_heights = sorted(w_h, reverse=True)

    elif len(w_h) > 2:
        list_of_height = w_h
        sorted_heights = sorted(w_h, reverse=True)
        highest_height_f = sorted_heights[0]
        avg_height = sum(sorted_heights) / len(sorted_heights)

        filtered_heights = [highest_height_f]
        filtered_heights += [h for h in sorted_heights[1:] if abs(highest_height_f - h) < 20]

    else:
        filtered_heights = w_h

    if verbose:
        logger.write('>' * 25 + f' BORDER: FILTER READTEXT FRAME ' + '>' * 25, logger.DEBUG)
        logger.write(f'LIST OF HEIGHT: {list_of_height}, SORTED HEIGHT: {sorted_heights}, FILTERED HEIGHTS: {filtered_heights}, AVG HEIGHT: {avg_height}', logger.DEBUG)

    return filtered_heights 

def save_cropped_images(self, cropped_images, save_dir="cropped_images"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for cropped_image in cropped_images:
        # Create a timestamped filename for saving the cropped image
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        image_path = os.path.join(save_dir, f"cropped_image_{timestamp}.png")
        cv2.imwrite(image_path, cropped_image)
        # print(f"Saved: {image_path}")