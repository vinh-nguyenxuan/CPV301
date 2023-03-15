import os
from stitiching import *
from PyQt5 import uic
from PIL import Image as im
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QFileDialog


#Vui lòng copy đường dẫn của file interface vào đây;
PATH = "D:/myCode/CPV301/Workshop6/interface.ui"

class interface(QMainWindow):
    def __init__(self):
        super(interface, self).__init__()
        uic.loadUi(PATH, self)
        self.show()

        self.image_1 = None
        self.image_2 = None
        self.image_3 = None

        #Setting button;
        self.pushButton.clicked.connect(self.linktoimage1)
        self.pushButton_2.clicked.connect(self.linktoimage2)
        self.pushButton_4.clicked.connect(self.linktoimage3)
        self.pushButton_3.clicked.connect(self.enter)

    def linktoimage1(self):
        self.image_1 = QFileDialog.getOpenFileName(filter='*.jpg *.png')
        self.label.setPixmap(QPixmap(self.image_1[0]))
        self.lineEdit.setText(self.image_1[0])

    def linktoimage2(self):
        self.image_2 = QFileDialog.getOpenFileName(filter='*.jpg *.png')
        self.label_2.setPixmap(QPixmap(self.image_2[0]))
        self.lineEdit_2.setText(self.image_2[0])

    def linktoimage3(self):
        self.image_3 = QFileDialog.getOpenFileName(filter='*.jpg *.png')
        self.label_3.setPixmap(QPixmap(self.image_3[0]))
        self.lineEdit_4.setText(self.image_3[0])

    def enter(self):
        imgs_path = [self.image_1[0], self.image_2[0], self.image_3[0]]
        imgres = stitching(imgs_path)
        cv.imwrite("image.png", imgres)
        self.label_7.setPixmap(QPixmap("image.png"))
        self.lineEdit_3.setText("Done......!")



if __name__ == "__main__":
    app = QApplication([])
    myprogram = interface()
    app.exec()

