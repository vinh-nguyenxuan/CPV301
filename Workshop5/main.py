import os
from ransac import *
from PyQt5 import uic
from PIL import Image as im
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QFileDialog

#Vui lòng copy đường dẫn của file interface vào đây;
PATH = "D:/myCode/CPV/Workshop5/interface.ui"

class interface(QMainWindow):
    def __init__(self):
        super(interface, self).__init__()
        uic.loadUi(PATH, self)
        self.show()

        self.image_ref = None
        self.image_ali = None

        #Setting button;
        self.pushButton.clicked.connect(self.linktoimage_ref)
        self.pushButton_2.clicked.connect(self.linktoimage_ali)
        self.pushButton_3.clicked.connect(self.enter)

    def linktoimage_ref(self):
        self.image_ref = QFileDialog.getOpenFileName(filter='*.jpg *.png')
        self.label.setPixmap(QPixmap(self.image_ref[0]))
        self.lineEdit.setText(self.image_ref[0])

    def linktoimage_ali(self):
        self.image_ali = QFileDialog.getOpenFileName(filter='*.jpg *.png')
        self.label_2.setPixmap(QPixmap(self.image_ali[0]))
        self.lineEdit_2.setText(self.image_ali[0])

    def enter(self):
        img1 = cv.imread(self.image_ref[0], cv.IMREAD_COLOR)
        img2 = cv.imread(self.image_ali[0], cv.IMREAD_COLOR)

        imgRes = alignImages(img2, img1)
        cv.imwrite("image.png", imgRes)
        
        self.label_3.setPixmap(QPixmap("image.png"))
        self.lineEdit_3.setText("Done......!")



if __name__ == "__main__":
    app = QApplication([])
    myprogram = interface()
    app.exec()

