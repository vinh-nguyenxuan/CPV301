import os
import data
import path
import joblib
import cv2 as cv
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

#Data Lableing;
def creat_label():
    res = {}
    final = {}

    file = open(path.path_students_in_class, "r")
    data = file.read()
    data = data.split("\n")
    data = data[:-1]

    count = 0
    for i in data:
        i = i.split("/")
        res[i[0]] = i[1]
        count += 1
    
    lst_id = []
    for i in os.listdir(path.path_train):
        lst_id.append(i)

    for i in range(len(lst_id)):
        final[i] = res[lst_id[i]]

    return final

#Training model;
def training():
    pca = PCA()
    X_train, y_train, X_test, y_test = data.creat_data_table(path.path_train, path.path_test)
    clf = LinearSVC(max_iter=100000)
    clf = CalibratedClassifierCV(clf)
    X_train = pca.fit_transform(X_train)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(pca.transform(X_test))
    print("Performance Of The Model")
    print(classification_report(y_test, y_predict))
    joblib.dump(clf, "D:/myCode/CPV301/Assignment/file_and_data/file/model.npy")

#rRun model;
def run():
    label = creat_label()
    face_detection = cv.CascadeClassifier(path.path_haar)
    capture = cv.VideoCapture(0)
    model = joblib.load(path.path_model)
     
    print(label)
    while True:

        pca = PCA()
        isTrue, frame = capture.read()
        model = joblib.load(path.path_model)
        faces = face_detection.detectMultiScale(frame)
        X_train, y_train, X_test, y_test = data.creat_data_table(path.path_train, path.path_test)

        x = pca.fit_transform(X_train)

        for (x, y, w, h) in faces:
            face = frame[y: h+y, x: w+x]
            face = cv.resize(face, (64, 64))
            face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
            face = data.image_preprocessing(face)
            face = [face]
            face = pca.transform(face)
            y_predict = model.predict(face)
            print(y_predict, y_predict[0])
            name = label[y_predict[0]]

            cv.putText(frame, name, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv.imshow("Webcam", frame)
        if cv.waitKey(1) & 0xFF == ord('d'):
            break


# run()
# training()
