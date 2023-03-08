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


#Function 4a;
def Function4():

    while True:
        theta = input("Enter The Degree You Want: ")
        try:
            theta = float(theta)
        except:
            theta = None
        if theta != None:
            theta = float(theta)
            break

    theta_input = theta
    theta = theta * math.pi / 180


    size_matrix = (20, 20)
    color = random.randint(0, 255)
    img_matrix = np.full((1, size_matrix[1]), color)
    
    for i in range(0, size_matrix[0] - 1):
        color = random.randint(0, 255)
        buffer_array = np.full((1, size_matrix[1]), color)
        img_matrix = np.concatenate((img_matrix, buffer_array))


    coords = []

    array_y = 0
    for y in range(int((size_matrix[0] - 1) / 2), int(-1 * (size_matrix[0] - 1) / 2), -1):
        array_x = 0
        for x in range(-1 * int((size_matrix[0] - 1) / 2), int((size_matrix[0] - 1) / 2)):
            coords.append((x, y, array_x, array_y))
            array_x += 1
        array_y += 1

    def rotation_function(theta, point):
        x, y, color_x, color_y = point

        point = (x, y)

        new_point = (round((point[0] * math.cos(theta) - point[1] * math.sin(theta)), 1), round((point[0] * math.sin(theta) + point[1] * math.cos(theta)), 1))

        return (new_point[0], new_point[1], color_x, color_y)
    
    def graph_points_all(points):
        all_x = []
        all_y = []
        colors = []
        for x, y, index_x, index_y in points:
            all_x.append(x)
            all_y.append(y)
            colors.append(img_matrix[coords[index_y][2]][index_x])

        plt.figure(figsize=(15, 15))
        plt.subplots_adjust(wspace=0.2, hspace=0.5)
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            if i == 0:
                plt.imshow(img_matrix)
                plt.title("The Original Image")
            else:
                plt.scatter(all_x, all_y, c=colors, s=17**2, marker='s')
                plt.title("The Image After Rotation Anticlockwise In " +str(theta_input) + " Degrees")
        plt.show()
    
    points = []  
    for point in coords:
        points.append(rotation_function(theta, point))

    graph_points_all(points)

Function4()