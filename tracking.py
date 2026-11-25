

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.track(
    source="people.mp4",
    tracker="botsort.yaml",
    show=True,
    persist=True
)
