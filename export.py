from ultralytics import YOLO

# Load model
model = YOLO("yolo11m.pt")

# Export the model
#model.export(format="engine", device="0", half=True, imgsz=640)
model.export(format="engine", device="0", half=True, imgsz=1280)