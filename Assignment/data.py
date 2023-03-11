import os
import path
import cv2 as cv
import numpy as np
import albumentations as alb
from skimage.feature import hog
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator

#Augmentation;
def augment_images(path_student):
    augmentor = alb.Compose([alb.HorizontalFlip(p=0.2),
                            alb.HorizontalFlip(p=0.3), 
                            alb.RandomBrightnessContrast(p=0.2),
                            alb.RandomGamma(p=0.2), 
                            alb.RGBShift(p=0.2),
                            alb.RGBShift(p=0.5)])

    for j in os.listdir(path_student):
        img_path = path_student + "/" + j
        img = cv.imread(img_path)
        for x in range(10):
            aug = augmentor(image=img)
            cv.imwrite(img_path[:-4] + str(x) + ".jpg", aug["image"])

#Image loading;
def load_images(path):
    dic = {}
    label = []
    for i in os.listdir(path):
        label.append(i)
        folder_path = path + "/" + i
        lst = []
        for j in os.listdir(folder_path):
            img_path = folder_path + "/" + j
            img = cv.imread(img_path, 0)
            img = cv.resize(img, (64, 64))
            img = img * (1./ 255)
            lst.append(img)
        dic[i] = lst
    return dic, label

#Preprocessing images;
def image_preprocessing(img):
    img = cv.GaussianBlur(img, (3, 3), 0)
    fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', feature_vector=True)
    return fd

#Image to vector;
def image_to_vector(data, label):
    lst_data = []
    label_data = []
    for i in data:
        for j in data[i]:

            j = image_preprocessing(j)
            label_j = label.index(i)
            lst_data.append(j)
            label_data.append(label_j)
    
    lst_data = np.array(lst_data)
    label_data = np.array(label_data)
    lst_data = np.concatenate((lst_data, label_data[:, None]), axis = 1)
    lst_data = shuffle(lst_data)
    return lst_data

#Split dataset;
def split_data(data):
    Y = data[:, -1]
    X = np.delete(data, -1, 1)
    return X, Y

#Dataset to data table;
def creat_data_table(path_train, path_test):

    data_train, label_train = load_images(path_train)
    data_test, label_test = load_images(path_test)

    train = image_to_vector(data_train, label_train)
    test = image_to_vector(data_test, label_test)

    X_train, y_train = split_data(train)
    X_test, y_test = split_data(test)

    return X_train, y_train, X_test, y_test





