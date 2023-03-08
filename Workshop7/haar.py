import cv2 as cv
face_detection = cv.CascadeClassifier("D:/myCode/CPV/Workshop8/haarcascade_frontalface_default.xml")

def haar(path):
    img = cv.imread(path)
    faces = face_detection.detectMultiScale(img)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img