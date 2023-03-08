import math
import random
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Variables global. 
draw = False
x1 = y1 = -1
check = False
h = k = None
x_old = y_old = None
check_move = False
lst = [] 
image = np.zeros((2000, 2000))

#Function 3;
def Function3(image):
    def Draw_A_Rectangle(event, x, y, flags, param):
        global draw, x1, y1, h, k, check, x_old, y_old
        if event == cv.EVENT_LBUTTONDOWN and check == False:
            x1 = x
            y1 = y
            draw = True
        elif event == cv.EVENT_MOUSEMOVE and draw == True:
            cv.rectangle(image, (x1, y1), (x, y), (255, 255, 255), -1)
        elif event == cv.EVENT_LBUTTONUP and draw == True:
            draw = False
            check = True
            cv.rectangle(image, (x1, y1), (x, y), (255, 255, 255), -1)
            h, k = (x - x1), (y - y1)
            x_old, y_old = x, y
        elif event == cv.EVENT_LBUTTONDOWN and check == True:
            cv.rectangle(image, (x1, y1), (x_old, y_old), (0, 0, 0), -1)
            cv.rectangle(image, (x, y), (x + h, y + k), (255, 255, 255), -1)
            x1, y1 = x, y
            x_old, y_old = x + h , y + k
            

    cv.namedWindow("Draw")
    cv.setMouseCallback("Draw", Draw_A_Rectangle)
    
    while True:

        cv.imshow("Draw", image)
        if cv.waitKey(1) & 0xFF == ord('d'):
            break
    cv.destroyAllWindows()

Function3(image)