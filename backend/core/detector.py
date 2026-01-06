from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")

def detect_objects(frame):
    return model(frame, stream=True)
