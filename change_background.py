import cv2
import numpy as np

# image_path = r"C:\Users\DOT\Documents\febri\bitbucket\bugfixAdd-new-Camera-Pos\ai-smartparking\plate_saved\2024-11-04-14-59-50-542237.jpg"
# image_path = r"C:\Users\DOT\Documents\febri\bitbucket\bugfixAdd-new-Camera-Pos\ai-smartparking\plate_saved\2024-11-04-14-59-51-208309.jpg"
image_path = r"C:\Users\DOT\Documents\febri\backup\2024-11-04\plate_saved\2024-11-04-14-59-50-542237.jpg"
image = cv2.imread(image_path)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_green = np.array([30, 40, 40])
upper_green = np.array([90, 255, 255])
# lower_green = np.array([35, 40, 40])
# upper_green = np.array([85, 255, 255])

green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

image[green_mask > 0] = [255, 255, 255]

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()