import os
from haar import *
from PyQt5 import uic
from PIL import Image as im
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QFileDialog

PATH = "D:/myCode/CPV/Workshop7/interface.ui"

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
        self.image = QFileDialog.getOpenFileName(filter='*.jpg *.png *.jpeg')
        self.label.setPixmap(QPixmap(self.image[0]))
        self.lineEdit.setText(self.image[0])

    def enter(self):
        img_res = haar(self.image[0])
        cv.imwrite("image.png", img_res)
        self.label_2.setPixmap(QPixmap("image.png"))
        self.lineEdit_2.setText("Done!")


if __name__ == "__main__":
    app = QApplication([])
    myprogram = interface()
    app.exec()