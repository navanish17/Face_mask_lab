import cv2
from ultralytics import YOLO

model = YOLO("best_face_mask.onnx")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, conf=0.2, iou=0.6)[0]
    cv2.imshow("Face Mask Detection", result.plot())

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
