import cv2
from ultralytics import YOLO

model = YOLO("model/best_face_mask.onnx")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)[0]
    cv2.imshow("Face Mask Detection", results.plot())

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
