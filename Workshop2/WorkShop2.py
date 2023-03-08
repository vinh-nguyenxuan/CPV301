import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Function1;
def check_color(str_):

    color = None

    while True:
        color = input(str_)
        try:
            color = float(color)
        except:
            color = -1000
        
        if color <= 255 and color >= -255:
            return color
####################################;
def color_balance(img):
    shape = img.shape

    img_org = img.copy()
    img_after = img.copy()

    stop_or_not = None
    while True:
        r = g = b = 0
        if len(shape) == 3:
            r = check_color("Enter the red number: ")
            g = check_color("Enter the green number: ")
            b = check_color("Enter the blue number: ")
            
            img_after[:, :, 0] = img_after[:, :, 0] + r
            img_after[:, :, 1] = img_after[:, :, 1] + g
            img_after[:, :, 2] = img_after[:, :, 2] + b
        else:
            g = check_color("Enter the color number: ")
            img_after = img_after + g

        lst = [img_org, img_after]
        plt.figure(figsize=(20, 15))
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            if i == 1:
                plt.title("Original Image")
            else:
                plt.title("Image After Color Balance")
            plt.imshow(cv.cvtColor(lst[i], cv.COLOR_BGR2RGB))
        plt.show()

        stop_or_not = input("Stop? [Y/N?]: ")
        if stop_or_not == 'Y' or stop_or_not == 'y':
            break
        img_org = img_after.copy()

#Function2;
def histogram_equalization(img):

    lst_histogram_org = [0 for i in range(256)]
    lst_histogram_new = [0 for i in range(256)]

    shape_ = img.shape

    for i in range(shape_[0]):
        for j in range(shape_[1]):
            lst_histogram_org[img[i][j]] += 1
    
    for i in range(256):
        lst_histogram_new[i] = sum(lst_histogram_org[: i])

    lst_histogram_new = ((lst_histogram_new - np.min(lst_histogram_new)) / (np.max(lst_histogram_new) - np.min(lst_histogram_new))) * 255

    for i in range(shape_[0]):
        for j in range(shape_[1]):
            img[i, j] = lst_histogram_new[img[i,j]]

    return img
####################################;
def visualization(lst):
    plt.figure(figsize=(20, 15))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        if i == 0 or i == 1:
            plt.imshow(cv.cvtColor(lst[i], cv.COLOR_BGR2RGB))
            if i == 0:
               plt.title("Original Image")
            else:
               plt.title("Image After Histogram Equalization")
        else:
            plt.plot(lst[i])
            if i == 2:
                plt.title("Histogram Original Image")
            else:
                plt.title("Histogram Image After Histogram Equalization")
    plt.show()
####################################;
def visualization2(lst):
    plt.figure(figsize=(20, 20))
    for i in range(8):
        plt.subplot(4, 2, i + 1)
        if i == 0 or i == 1:
            plt.imshow(cv.cvtColor(lst[i], cv.COLOR_BGR2RGB))
            if i == 0:
               plt.title("Original Image")
            else:
               plt.title("Image After Histogram Equalization")
        elif i==2 or i == 3:
            plt.plot(lst[i])
            if i == 2:
                plt.title("Histogram Original Image R")
            else:
                plt.title("Histogram Image After Histogram Equalization R")
        elif i==4 or i == 5:
            plt.plot(lst[i])
            if i == 4:
                plt.title("Histogram Original Image G")
            else:
                plt.title("Histogram Image After Histogram Equalization G")
        elif i==6 or i == 7:
            plt.plot(lst[i])
            if i == 6:
                plt.title("Histogram Original Image B")
            else:
                plt.title("Histogram Image After Histogram Equalization B")
    plt.show()
####################################;
def function2(img):
    img_org = img.copy()
    img_after = img.copy()
    hist_org = cv.calcHist([img_org], [0], None, [255], [0, 255])

    shape = img.shape
    
    if len(shape) == 3:
        for i in range(3):
            img_after[:, :, i] = histogram_equalization(img_after[:, :, i])
    else:
        img_after = histogram_equalization(img_after)

    hist_after = cv.calcHist([img_after], [0], None, [255], [0, 255])
    if len(shape) == 2:
        lst = [img_org, img_after, hist_org, hist_after]
        visualization(lst)
    else:
        hist_org_r = cv.calcHist([img_org[:, :, 0]], [0], None, [255], [0, 255])
        hist_org_g = cv.calcHist([img_org[:, :, 1]], [0], None, [255], [0, 255])
        hist_org_b = cv.calcHist([img_org[:, :, 2]], [0], None, [255], [0, 255])

        hist_after_r = cv.calcHist([img_after[:, :, 0]], [0], None, [255], [0, 255])
        hist_after_g = cv.calcHist([img_after[:, :, 1]], [0], None, [255], [0, 255])
        hist_after_b = cv.calcHist([img_after[:, :, 2]], [0], None, [255], [0, 255])

        lst = [img_org, img_after, hist_org_r, hist_after_r, hist_org_g, hist_after_g, hist_org_b, hist_after_b]
        visualization2(lst)

#Function 3, 4;
class Filter_Median_And_Mean():
    
    def __init__(self, image, kernal_size, pad, stride, type_conv):
        
        self.image = image
        self.keranl_size = kernal_size
        self.pad = pad
        self.stride = stride
        self.type = type_conv

        self.W_image = self.image.shape[0]
        self.H_image = self.image.shape[1]


        self.W = int((self.W_image + (2 * self.pad) - kernal_size) / self.stride) + 1
        self.H = int((self.H_image + (2 * self.pad) - kernal_size) / self.stride) + 1

        self.image_res = image.copy()

    def padding(self):
        image_padding = np.pad(self.image, ((self.pad, self.pad), (self.pad, self.pad)), 'constant', constant_values = 0)
        return image_padding

    def value(self, sub_matrix):
        
        if self.type == "Median":
            sub_matrix = sub_matrix.flatten()
            sub_matrix = sorted(sub_matrix)
            median = len(sub_matrix) // 2
            return sub_matrix[median]
        return np.mean(sub_matrix)

    def main(self):
        
        self.image = self.padding()
        for w in range(self.W):    
            w_start = w * self.stride
            w_end = w_start + self.keranl_size
            for h in range(self.H):
                h_start = h * self.stride
                h_end = h_start + self.keranl_size
                shape = self.image_res[w_start: w_end, h_start: h_end].shape
                self.image_res[w_start: w_end, h_start: h_end][shape[0] // 2][shape[1] // 2] = self.value(self.image_res[w_start: w_end, h_start: h_end])
####################################;
def input_pram(img):  
    kernal_size = None
    while True:
        kernal_size = input("Enter your kernal size you want: ")
        try:
            kernal_size = int(kernal_size)
        except:
            kernal_size = -1
        
        if kernal_size >= 0 and kernal_size <= img.shape[0] and kernal_size % 2 == 1:
            return kernal_size
####################################;
def function3(img):
    img_org = img.copy()
    img_after = img.copy()

    shape = img.shape
    kernal_size = input_pram(img)
    padding = (kernal_size - 1) // 2
    stride = 1
    if len(shape) == 3:
        for i in range(3):
            pivot_img = Filter_Median_And_Mean(img_after[:, :, i], kernal_size, padding, stride, "Median")
            pivot_img.main()
            img_after[:, :, i] = pivot_img.image_res
    else:
        pivot_img = Filter_Median_And_Mean(img_after, 3, 1, 1, "Median")
        pivot_img.main()
        img_after = pivot_img.image_res

    plt.figure(figsize=(20, 15))
    lst = [img_org, img_after]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(cv.cvtColor(lst[i], cv.COLOR_BGR2RGB))
        if i == 1:
            plt.title("Image After Use The Median Filter")
        else:
            plt.title("Original Image")
    plt.show()
####################################;
def function4(img):

    img_org = img.copy()
    img_after = img.copy()

    shape = img.shape
    kernal_size = input_pram(img)
    padding = (kernal_size - 1) // 2
    stride = 1
    if len(shape) == 3:
        for i in range(3):
            pivot_img = Filter_Median_And_Mean(img_after[:, :, i], kernal_size, padding, stride, "Mean")
            pivot_img.main()
            img_after[:, :, i] = pivot_img.image_res
    else:
        pivot_img = Filter_Median_And_Mean(img_after, 3, 1, 1, "Mean")
        pivot_img.main()
        img_after = pivot_img.image_res

    plt.figure(figsize=(20, 15))
    lst = [img_org, img_after]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(cv.cvtColor(lst[i], cv.COLOR_BGR2RGB))
        if i == 1:
            plt.title("Image After Use The Mean Filter")
        else:
            plt.title("Original Image")
    plt.show()

#Function5;
class Filter_Gaussian():
    
    def __init__(self, image, kernal, pad, stride):
        
        self.image = image
        self.keranl = kernal
        self.pad = pad
        self.stride = stride

        self.W_image = self.image.shape[0]
        self.H_image = self.image.shape[1]

        self.W = int((self.W_image + (2 * self.pad) - self.keranl.shape[0]) / self.stride) + 1
        self.H = int((self.H_image + (2 * self.pad) - self.keranl.shape[1]) / self.stride) + 1

        self.image_res = image.copy()

    def padding(self):
        image_padding = np.pad(self.image, ((self.pad, self.pad), (self.pad, self.pad)), 'constant', constant_values = 0)
        return image_padding

    def conv(self, w_start, w_end, h_start, h_end):
        
        s = np.multiply(self.image[w_start: w_end, h_start: h_end], self.keranl)
        S = np.sum(s)
        return S

    def main(self):
        
        self.image = self.padding()
        for w in range(self.W):    
            w_start = w * self.stride
            w_end = w_start + self.keranl.shape[0]
            for h in range(self.H):
                h_start = h * self.stride
                h_end = h_start + self.keranl.shape[1]
                shape = self.image_res[w_start: w_end, h_start: h_end].shape
                self.image_res[w_start: w_end, h_start: h_end][shape[0] // 2][shape[1] // 2] = self.conv(w_start, w_end, h_start, h_end)
####################################;
def input_pram5(img):  
    kernal_size = None
    while True:
        kernal_size = input("Enter your kernal size you want: ")
        try:
            kernal_size = int(kernal_size)
        except:
            kernal_size = -1
        
        if kernal_size >= 0 and kernal_size <= img.shape[0] and kernal_size % 2 == 1 and kernal_size in [3, 5, 7]:
            return kernal_size
####################################;
def filter_gaussian(kerna_size):
    
    if kerna_size == 3:
        matrix_3 = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        return matrix_3
    
    if kerna_size == 5:
        matrix_5 = 1 / 273 * np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])
        return matrix_5
    
    if kerna_size == 7:
        matrix_7 = 1 / 1003 * np.array([[0, 0, 1, 2, 1, 0, 0], [0, 3, 13, 22, 13, 3, 0], [1, 13, 59, 97, 59, 13, 1], [2, 22, 97, 159, 97, 22, 2], [1, 13, 59, 97, 59, 13, 1], [0, 3, 13, 22, 13, 3, 0], [0, 0, 1, 2, 1, 0, 0]])
        return matrix_7
####################################;
def function5(img):

    img_org = img.copy()
    img_after = img.copy()

    shape = img.shape
    kernal_size = input_pram(img)
    padding = (kernal_size - 1) // 2
    stride = 1
    kernal = filter_gaussian(kernal_size)
    if len(shape) == 3:
        for i in range(3):
            pivot_img = Filter_Gaussian(img_after[:, :, i], kernal, padding, stride)
            pivot_img.main()
            img_after[:, :, i] = pivot_img.image_res
    else:
        pivot_img = Filter_Gaussian(img_after, 3, 1, 1)
        pivot_img.main()
        img_after = pivot_img.image_res

    plt.figure(figsize=(20, 15))
    lst = [img_org, img_after]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(cv.cvtColor(lst[i], cv.COLOR_BGR2RGB))
        if i == 1:
            plt.title("Image After Use The Gaussian Filter")
        else:
            plt.title("Original Image")
    plt.show()

#Function menu;
def menu():

    print("================= MENU =================")
    print("                                        ")
    print("Function1: Color Balance                ")
    print("Function2: Histogram Equalization       ")
    print("Function3: Implement The Median Filter  ")
    print("Function4: Implement The Mean Filter    ")
    print("Function5: Implement Gaussian           ")
    print("Function6: Quit                         ")
    print("                                        ")
####################################;
def check_input_menu():
    
    while True:
        selection = input("Enter your selection: ")
        try:
            selection = int(selection)
        except:
            selection = -1
        
        if selection in [i for i in range(1, 7)]:
            break
    
    return selection
####################################;
def load_image(img_path):
    channel = None

    while True:
        channel = input("Enter your channel you want: ")
        try:
            channel = int(channel)
        except:
            channel = -1
        if channel == 1 or channel == 3:
            break

    if channel == 1:
       img = cv.imread(img_path, 0)
    else:
        img = cv.imread(img_path)
    img = cv.resize(img, (600, 600))
    return img
####################################;
def calculate_padding(kernal_size):
    P = (kernal_size - 1) // 2
    return P 
####################################;
def selection_function(selection, img):

    if selection == 1:
        color_balance(img)
    elif selection == 2:
        function2(img)
    elif selection == 3:
        function3(img)
    elif selection == 4:
        function4(img)
    elif selection == 5:
        function5(img)
    elif selection == 6:
        print("Thank You!")
    else:
        print("Invalid selection!")
####################################;
def res_your_selection(img_path):

    img = load_image(img_path)

    while True:

        menu()
        selection = check_input_menu()
        selection_function(selection, img)

        if selection == 6:
            break

def load_path():
    path = input("Enter image path: ")
    new_path = ""
    for i in path:
        if ord(i) == 92:
            new_path += "/"
        else:
            new_path += i
    return new_path

def check_path_img():
    while True:
        check = 0
        img_path = load_path()
        try:
            img = cv.imread(img_path[1:-1])
        except:
            check += 1
        
        if check == 0:
            return img_path

####################################;
def main():
    img_path = check_path_img()
    res_your_selection(img_path[1: -1])

#main;
if __name__ == "__main__":
    main()




