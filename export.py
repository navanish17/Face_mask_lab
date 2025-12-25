from ultralytics import YOLO

model = YOLO("best_face_mask.pt")
model.export(format="onnx")
