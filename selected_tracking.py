
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

selected_id = int(input("Enter the object ID: "))

video = cv2.VideoCapture("people.mp4")

while True:

    ret, frame = video.read()

    if not ret:
        break

    # Track objects in each frame
    results = model.track(
        frame,
        tracker="botsort.yaml",
        persist=True,
        verbose=False
    )

    result = results[0]

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.int().cpu().tolist()
        ids = result.boxes.id.int().cpu().tolist()

        for box, obj_id in zip(boxes, ids):

            if obj_id == selected_id:

                x1, y1, x2, y2 = box

                # Draw selected object bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Selected ID: " + str(obj_id),
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Selected Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
