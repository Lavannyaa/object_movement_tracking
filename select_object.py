
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Ask user to select an object ID
selected_id = int(input("Enter the object ID to select: "))

video_path = "people.mp4"

# Track objects in video
results = model.track(
    source=video_path,
    tracker="botsort.yaml",
    stream=True,
    persist=True
)

for result in results:

    if result.boxes.id is not None:

        ids = result.boxes.id.int().tolist()

        for i, obj_id in enumerate(ids):

            if obj_id == selected_id:
                print("Selected Object ID:", obj_id)
