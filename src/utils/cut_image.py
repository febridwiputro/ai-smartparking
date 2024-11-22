import os
import cv2
from src.Integration.arduino import Arduino
from src.view.show_cam import resize_image, show_cam
from src.controller.ocr_controller import OCRController


def main():
    input_vid = r"C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\keluar_lt_2_out.mp4"
    cap = cv2.VideoCapture(input_vid)
    if not cap.isOpened():
        print("Cannot open video")
        exit()
    
    arduino = [Arduino(baudrate=115200, driver="CP210") for _ in range(1, 3)]
    plat_detect = OCRController(arduino, "asd")
    tambah = 0
    
    # Define output directories
    output_dir1 = r"C:\Users\DOT\Documents\ai-smartparking\exp\aset_keluars"
    output_dir2 = r"C:\Users\DOT\Documents\ai-smartparking\exp\aset_keluar_processings"
    
    # Create directories if they don't exist
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            tambah += 1
            text, f, plat, preplat = plat_detect._processing_ocr(frame)
            cv2.imwrite(os.path.join(output_dir1, f"plat_{text}_{tambah}.jpg"), plat)
            cv2.imwrite(os.path.join(output_dir2, f"plat_{text}_{tambah}.jpg"), plat)
        
        show_cam("frame", f)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
