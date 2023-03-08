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

#Function 1; 
def Function1():
    background = np.zeros((255, 255))
    background += 255
    cv.imshow("While BackGround", background)
    cv.waitKey(0)
    cv.destroyAllWindows()

Function1()