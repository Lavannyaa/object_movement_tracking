from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

img = cv2.imread("frames/frame_0.jpg")

results = model(img)

output = results[0].plot()

cv2.imshow("YOLO Detection", output)

cv2.waitKey(0)
cv2.destroyAllWindows()

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])

        object_name = model.names[class_id]

        print("Detected Object:", object_name)
