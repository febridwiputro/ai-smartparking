import cv2
import os
import argparse

def spliting_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Membuat folder jika belum ada
    folder_name = "frames"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.convertScaleAbs(frame_hsv, frame_hsv, 1.0, 1.0)
        # Setiap 1 detik, dapatkan 10 frame
        if frame_count % 10 == 0:
            cv2.imwrite(os.path.join(folder_name, f"frame_{frame_count:04d}.jpg"), frame)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the image file")
    args = vars(ap.parse_args())
    spliting_video(args["image"])