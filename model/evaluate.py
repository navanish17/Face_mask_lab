from ultralytics import YOLO

def evaluate():
    model = YOLO("best.pt")
    model.val(data="mask.yaml")

if __name__ == "__main__":
    evaluate()
