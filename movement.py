
from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Get selected object ID
selected_id = int(input("Enter the object ID: "))

# Open video
video = cv2.VideoCapture("people.mp4")

previous_x = None
previous_y = None

while True:

    ret, frame = video.read()

    if not ret:
        break

    # Detect and track objects
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

                # Calculate centre coordinates
                current_x = (x1 + x2) // 2
                current_y = (y1 + y2) // 2

                if previous_x is not None:

                    # Calculate displacement
                    dx = current_x - previous_x
                    dy = current_y - previous_y

                    print("X Displacement:", dx, "pixels")
                    print("Y Displacement:", dy, "pixels")

                    # Detect movement direction
                    if abs(dx) < 3 and abs(dy) < 3:
                        direction = "Stationary"

                    elif abs(dx) > abs(dy):
                        if dx > 0:
                            direction = "Right"
                        else:
                            direction = "Left"

                    else:
                        if dy > 0:
                            direction = "Down"
                        else:
                            direction = "Up"

                    print("Direction:", direction)
                    print("-------------------")

                # Update previous position
                previous_x = current_x
                previous_y = current_y

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw centre point
                cv2.circle(
                    frame,
                    (current_x, current_y),
                    5,
                    (0, 0, 255),
                    -1
                )

                # Display direction on video
                if previous_x is not None:
                    cv2.putText(
                        frame,
                        "Selected ID: " + str(selected_id),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

    cv2.imshow("Movement Direction Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
