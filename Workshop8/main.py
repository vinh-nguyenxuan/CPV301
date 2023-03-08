import os
from eigenfaces import *
from PyQt5 import uic
from PIL import Image as im
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QFileDialog


#Vui lòng copy đường dẫn của file interface vào đây;
PATH = "D:/myCode/CPV/Workshop8/interface.ui"
face_detection = cv.CascadeClassifier("D:/myCode/CPV/Workshop8/haarcascade_frontalface_default.xml")

#Tag name;
names = {0: "TuKha", 1: "HoangAnh", 2: "NhatTruong", 3: "PhucTho"}

class interface(QMainWindow):
    def __init__(self):
        super(interface, self).__init__()
        uic.loadUi(PATH, self)
        self.show()

        self.image = None

        # Setting button;
        self.pushButton.clicked.connect(self.linktoimage)
        self.pushButton_3.clicked.connect(self.enter)

    def linktoimage(self):
        self.image = QFileDialog.getOpenFileName(filter='*.jpg *.png')
        self.label.setPixmap(QPixmap(self.image[0]))
        self.lineEdit.setText(self.image[0])

    def enter(self):
        predict = run(self.image[0])
        faces = face_detection.detectMultiScale(cv.imread(self.image[0]))
        (x, y, w, h) = faces[0]
        img_new = cv.rectangle(cv.imread(self.image[0]), (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv.imwrite("image.png", img_new)
        self.label_2.setPixmap(QPixmap("image.png"))
        self.lineEdit_2.setText(names[predict])




if __name__ == "__main__":
    app = QApplication([])
    myprogram = interface()
    app.exec()

