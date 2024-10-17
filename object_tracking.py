import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"C:\Users\DOT\Documents\febri\weights\yolo11n.pt")

# Video path and setup
video_path = r'C:\Users\DOT\Documents\ai-smartparking\src\Assets\ocr_assets\z.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        frame = cv2.resize(frame, (1080, 720))
        results = model.track(frame, persist=True, classes=[2])
        
        for r in results[0].boxes:
            if r.id is not None and r.cls is not None and r.conf is not None:
                object_id = int(r.id.item())
                class_id = int(r.cls.item())
                bbox = r.xyxy[0].cpu().numpy().tolist()
                confidence = float(r.conf.item())

                print(f"Object ID: {object_id}, Class ID: {class_id}, Confidence: {confidence:.2f}, BBox: {bbox}")

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, f"ID: {object_id}, Conf: {confidence:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Tracking", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
