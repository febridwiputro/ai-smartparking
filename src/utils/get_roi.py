import cv2
import matplotlib.pyplot as plt
import argparse
import asyncio
import time
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.Integration.api import api


class LabelVideo:
    def __init__(self, video_path):
        self.points = []
        self.polygons = []
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.img = None


    def get_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left mouse button down event
            self.points.append((x, y))
            cv2.circle(self.img, (x, y), 5, (0, 255, 0), -1)  # Draw a circle at the point
            if len(self.points) > 1:
                cv2.line(self.img, self.points[-2], self.points[-1], (255, 0, 0), 2)  # Draw a line between points
            cv2.imshow("label", self.img)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right mouse button down event
            if len(self.points) > 2:  # Changed to 3 to complete a polygon (a minimum of 3 points)
                self.polygons.append(self.points.copy())
                self.draw_polygons()  # Redraw all polygons including the new one
                self.points = []
                print("Polygon completed")
            else:
                print("Not enough points to form a polygon")

        elif event == cv2.EVENT_MBUTTONDOWN:
            # Middle mouse button down event
            if self.points:
                self.points.pop()
                self.draw_polygons()  # Redraw the polygons after removing a point
            else:
                print("No points to remove")

    def draw_polygons(self):
        for poly in self.polygons:
            cv2.polylines(self.img, [np.array(poly)], isClosed=True, color=(0, 255, 255), thickness=2)
        if self.points:
            cv2.polylines(self.img, [np.array(self.points)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.imshow("label", self.img)

    def running(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                break
            key = cv2.waitKey(10) & 0xFF
            frame = cv2.resize(frame, (1080, 720))

            if self.polygons:
                for polygon_points in self.polygons:
                    for i in range(len(polygon_points)):
                        cv2.circle(frame, polygon_points[i], 5, (0, 255, 0), -1)
                        if i > 0:
                            cv2.line(frame, polygon_points[i - 1], polygon_points[i], (255, 0, 0), 2)
                if self.points:
                    for i in range(len(self.points)):
                        cv2.circle(frame, self.points[i], 5, (0, 255, 0), -1)
                        if i > 0:
                            cv2.line(frame, self.points[i - 1], self.points[i], (255, 0, 0), 2)

            cv2.imshow("label", frame)

            if key == ord("c"):
                print("Cancel Operation to label image")
                cv2.destroyAllWindows()
                self.cap.release()
                exit()

            if key == ord("q"):
                self.img = frame.copy()
                break

            elif key == 32:
                self.img = frame.copy()
                cv2.setMouseCallback("label", self.get_points)
                cv2.waitKey(0)
                cv2.setMouseCallback("label", lambda *args: None)

            elif key == ord('m'):
                if self.points:
                    self.points.pop()



        cv2.destroyAllWindows()

    def show_plt(self):
        if self.img is not None:
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            for polygon_points in self.polygons:
                polygon = plt.Polygon(polygon_points, closed=True, fill=None, edgecolor='r')
                plt.gca().add_patch(polygon)
            plt.title("Polygons")
            plt.show()
        else:
            print("No image to display")

    def convert_all_bbox(self):
        if self.img is None:
            return
        height, width, _ = self.img.shape
        for i, pol in enumerate(self.polygons):
            for index, bbox in enumerate(pol):
                self.polygons[i][index] = (bbox[0] / width, bbox[1] / height)

        print(self.polygons)

    async def main(self, area, cam_link, slot):
        tasks = []
        for i, pol in enumerate(self.polygons):
            for index, bbox in enumerate(pol):
                task = api.insert_bounding_box(
                    slot=slot + i,
                    area=area,
                    cam_link=cam_link,
                    type=index + 1,
                    data_bounding=f"{bbox[0]},{bbox[1]}"
                )
                tasks.append(task)
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("-i", "--image", required=True, help="Path to video")
    arg.add_argument('-a', '--area', required=True, help="Area for API")
    arg.add_argument('-c', '--cam_link', required=True, help="Camera link")
    arg.add_argument('-s', '--slot', required=False, help="Slot for starter", default=1, type=int)
    args = vars(arg.parse_args())
    label = LabelVideo(args["image"])
    label.running()
    label.show_plt()
    label.convert_all_bbox()
    rn = time.time()
    asyncio.run(label.main(area=args["area"], cam_link=args["cam_link"], slot=args["slot"]))
    print(f"Time taken: {time.time() - rn}")
