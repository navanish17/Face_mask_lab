from ultralytics import YOLO

def export_model():
    model = YOLO("best.pt")
    model.export(format="onnx")
    model.export(format="tflite")

if __name__ == "__main__":
    export_model()
