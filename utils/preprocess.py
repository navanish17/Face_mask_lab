import cv2

def preprocess_image(img, size=640):
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    return img
