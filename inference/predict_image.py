import cv2
from ultralytics import YOLO

model = YOLO("best.onnx")

img = cv2.imread("test.jpg")
results = model(img, conf=0.4)

cv2.imshow("Prediction", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()