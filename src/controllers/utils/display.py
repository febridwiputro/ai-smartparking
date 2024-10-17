import cv2

def draw_box(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        color = (255, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
    
    return frame