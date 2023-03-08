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
image = np.zeros((1000, 1000))


#Function 5a.
def Function5(image): 
    number_scale = None
    while True:
        type_scale = input("0: Zoom in.\n1: Zoom out.\nNhap lua chon cua ban: ")
        number_scale = input("Enter The Number Scale You Want: ")
        try:
            number_scale = float(number_scale)
            type_scale = int(type_scale)
        except:
            number_scale = None
            type_scale == None
        if number_scale != None and type_scale != None and type_scale in [0, 1]:
            number_scale = float(number_scale)
            type_scale = int(type_scale)
            break 

    def Draw_A_Rectangle(event, x, y, flags, param):
        global draw, x1, y1, h, k, x_old, y_old, lst, check
        if event == cv.EVENT_LBUTTONDOWN and check == False:
            x1 = x
            y1 = y
            lst.append(x1)
            lst.append(y1)
            draw = True
        elif event == cv.EVENT_MOUSEMOVE and draw == True:
            cv.rectangle(image, (x1, y1), (x, y), (255, 255, 255), -1)
        elif event == cv.EVENT_LBUTTONUP and draw == True:
            draw = False
            check = True
            cv.rectangle(image, (x1, y1), (x, y), (255, 255, 255), -1)
            x_old, y_old = x, y
            lst.append(x_old)
            lst.append(y_old)
        elif event == cv.EVENT_LBUTTONDOWN and check == True:
            cv.rectangle(image, (x1, y1), (x_old, y_old), (0, 0, 0), -1)
            if number_scale == 0 or number_scale == 1:
                cv.rectangle(image, (x1, y1), (x_old, y_old), (255, 255, 255), -1)
            if number_scale != 0 and type_scale == 0 and number_scale != 1:
                print("a")
                cv.rectangle(image, (lst[0], lst[1]), (int(lst[2] * number_scale - lst[0]), int(lst[3] * number_scale - lst[1])), (255, 255, 255), -1)
            if number_scale != 0 and type_scale == 1 and number_scale != 1:
                cv.rectangle(image, (lst[0], lst[1]), (int(lst[2] / number_scale + lst[0] / number_scale), int(lst[3] / number_scale + lst[0] / number_scale)), (255, 255, 255), -1)

    cv.namedWindow("Draw")
    cv.setMouseCallback("Draw", Draw_A_Rectangle)
    
    while True:

        cv.imshow("Draw", image)
        if cv.waitKey(1) & 0xff == ord('d'):
            break
    cv.destroyAllWindows()

Function5(image)