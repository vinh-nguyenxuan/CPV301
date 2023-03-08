import os
import joblib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

#SETUP;
RESIZE = (128, 128)
labels = {"SE160885": 0, "SE160891": 1, "SE161235": 2, "SE170056": 3}

#Path;
data_train_path = "D:/myCode/CPV/Workshop8/data/train"
data_test_path = "D:/myCode/CPV/Workshop8/data/test"
model_path = "D:/myCode/CPV/Workshop8/model.npy"

#Data Loading And Preprocessing;
def load_img_path(folder_data_path):
    lst_path = []
    for i in os.listdir(folder_data_path):
        folder_img = folder_data_path + "/" + i
        for j in os.listdir(folder_img):
            img_path = folder_img + "/" + j
            lst_path.append(img_path)
    return lst_path

def img_to_flatten(lst_path):

    data = []
    label = []

    for i in lst_path:
        
        #Label;
        if "SE160885" in i: label.append(0)
        elif "SE160891" in i: label.append(1)
        elif "SE161235" in i: label.append(2)
        else: label.append(3)

        #Data;
        img = cv.imread(i, 0)
        img = cv.GaussianBlur(img, (5, 5), 0)
        img = img / 255.0
        img = cv.resize(img, RESIZE)
        img = img.flatten()
        data.append(img)

    data = np.array(data)
    label = np.array(label)
    data = np.concatenate((data, label[:, None]), axis = 1)
    data = shuffle(data) 
    return data

def train_and_test():
    data_train_lst_path = load_img_path(data_train_path)
    data_train_lst = img_to_flatten(data_train_lst_path)

    data_test_lst_path = load_img_path(data_test_path)
    data_test_lst = img_to_flatten(data_test_lst_path)

    y_train = data_train_lst[:, -1]
    X_train = np.delete(data_train_lst, -1, 1)

    y_test = data_test_lst[:, -1]
    X_test = np.delete(data_test_lst, -1, 1)
    
    return X_train, y_train, X_test, y_test

def preprocessing(path):

    img = cv.imread(path, 0)
    img = cv.GaussianBlur(img, (5, 5), 0)
    img = img / 255.0
    img = cv.resize(img, RESIZE)

    img_copy = img.copy()
    img_copy = img_copy.flatten()


    data_val = [img_copy]
    data_val = np.array(data_val)
    
    return data_val, img

def training_model():
    pca = PCA()
    X_train, y_train, X_test, y_test = train_and_test()
    clf = LinearSVC(max_iter=100000)
    clf = CalibratedClassifierCV(clf)
    X_train = pca.fit_transform(X_train)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(pca.transform(X_test))
    print(classification_report(y_test, y_predict))
    joblib.dump(clf, "model.npy")

def run(path):
    X_train, y_train, X_test, y_test = train_and_test()
    model = joblib.load(model_path)
    data_val, img = preprocessing(path)
    pca = PCA()
    x = pca.fit_transform(X_train)
    data_val = pca.transform(data_val)
    predict = int(model.predict(data_val))

    return predict


