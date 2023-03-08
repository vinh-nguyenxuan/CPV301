import os
import data
import path
import joblib
import cv2 as cv
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

#Gắn nhãn cho dữ liệu;
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

#Huấn luyện mô hình;
def training():
    X_train, y_train, X_test, y_test = data.creat_data_table(path.path_train, path.path_test)
    clf = LinearSVC(max_iter=100000)
    clf = CalibratedClassifierCV(clf) 
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print(classification_report(y_test, y_predict))
    joblib.dump(clf, "D:/Assignment/file_and_data/file/model.npy")

#Khởi chạy mô hình;
def run():
    label = creat_label()
    face_detection = cv.CascadeClassifier(path.path_haar)
    capture = cv.VideoCapture(0)
    model = joblib.load(path.path_model)

    while True:

        isTrue, frame = capture.read()
        faces = face_detection.detectMultiScale(frame)
        model = joblib.load(path.path_model)

        for (x, y, w, h) in faces:
            face = frame[y: h+y, x: w+x]
            face = cv.resize(face, (64, 64))
            face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
            face = data.image_preprocessing(face)
            face = [face]
            y_predict_lst = model.predict_proba(face)
            max_proba = y_predict_lst[0].max()
            name = None
            if max_proba >= 0.6:
                label_index = list(y_predict_lst[0]).index(max_proba)
                name = label[label_index]
            else:
                name = "Unknown"

            cv.putText(frame, name, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv.imshow("Webcam", frame)
        if cv.waitKey(1) & 0xFF == ord('d'):
            break


run()
# training()
