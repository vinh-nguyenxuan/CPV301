import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Function1;
class Harris():

    def __init__(self, image, blocksize, sobelsize, k) -> None:
        self.image = image
        self.image_copy = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY)
        self.bocksize = (blocksize - 1) // 2
        self.neighborhoods = [i for i in range(self.bocksize * (-1), self.bocksize + 1)]
        self.sobel_size = sobelsize
        self.k = k

    def max_pixel(self, R, i, j):
        max_point = 0
        for k in [-1, 0, 1]:
            for l in [-1, 0, 1]:
                if (max_point < R[i+k][j+l]):
                    max_point = R[i+k][j+l]
        return max_point

    def multiply(self, I_x, I_y):
        (h, w) = I_x.shape
        result = np.empty((h, w))
        for i in range(0, h):
            for j in range(0, w):
                result[i][j] = I_x[i][j] * I_y[i][j]
        return result

    def nms(self, R, i, j, max_point):
        for k in [-1, 0, 1]:
            for l in [-1, 0, 1]:
                if (R[i+k][j+l] == max_point):
                    continue
                else:
                    R[i+k][j+l] = 0
    
    def trace(self, A, B):
        return A + B
        
    def det(self, A, B, C):
        det = self.multiply(A, B) - self.multiply(C, C)
        return det

    def main(self):

        self.image_copy = cv.GaussianBlur(self.image_copy, (5, 5), 2)
        GX = cv.Sobel(self.image_copy, cv.CV_64F, 1, 0, ksize=5)
        GY = cv.Sobel(self.image_copy, cv.CV_64F, 0, 1, ksize=5)

        imgX = self.multiply(GX, GX)
        imgXY = self.multiply(GX, GY)
        imgY = self.multiply(GY, GY)

        imgX = cv.GaussianBlur(imgX, (5, 5), 2)
        imgXY = cv.GaussianBlur(imgXY, (5, 5), 2)
        imgY = cv.GaussianBlur(imgY, (5, 5), 2)

        imgR = self.det(imgX, imgXY, imgY) - self.k * self.trace(imgX, imgXY)
        _, imgR  = cv.threshold(imgR , imgR .max()//500, imgR.max(), cv.THRESH_BINARY)
        (h, w) = imgR .shape

        for i in range(1, h - 1, 2):
            for j in range(1, w - 1, 2):
                max_pixel = self.max_pixel(imgR , i, j)
                self.nms(imgR , i, j, max_pixel)

        imgR_dst = cv.dilate(imgR , None)
        self.image[imgR_dst > 0.001*imgR_dst.max()] = (0, 0, 255)

#Function2;
class HOG():

    def __init__(self, image) -> None:
        self.image = cv.resize(image, (600, 600))
        self.image_ = None
        self.image_1 = None
        self.angle_map = None
        self.image_res = None

    def Gradients(self):

        dx = cv.Sobel(self.image, cv.CV_64F, 1, 0, ksize=1)
        dy = cv.Sobel(self.image, cv.CV_64F, 0, 1, ksize=1)
        mag, angle = cv.cartToPolar(dx, -dy, angleInDegrees=True)

        angle = angle%180
        angle_map = None

        if(len(self.image.shape)==3):
            h,w,c = self.image.shape
            angle_map = np.zeros((h, w, 3, 9)) 
        else:
            h,w = self.image.shape
            angle_map = np.zeros((h, w, 1, 9))
            mag = mag[:, :, None] 
            angle = angle[:, :, None] 
            
        mask = angle>160
        angle_map[:, :, :, 0] += (mag*mask)*(1-(180-mask*angle)/20)
        mask = angle<=20
        angle_map[:, :, :, 0] += (mag*mask)*(1-mask*angle/20)

        for i in range(1,9):
            mask = np.bitwise_and(angle>(i-1)*20,angle<=(i+1)*20)
            angle_map[:, :, :, i] += (mag*mask)*(1-np.abs((i*20-mask*angle)/20))
        self.angle_map = angle_map

    def _HOGinCell(self, cell_size=(8,8)):
        h,w,c,_ = self.angle_map.shape
        h2 = int(h/cell_size[0])
        w2 = int(w/cell_size[1])
        cell_hist = np.zeros((h2, w2, c, 9))
        tmp = np.zeros((h2, w, c, 9))
        for i in range(h2):
            tmp[i] = np.sum(self.angle_map[i*cell_size[0]: (i+1)*cell_size[0]], axis=0)
        for i in range(w2):
            cell_hist[:,i] = np.sum(tmp[:, i*cell_size[1]: (i+1)*cell_size[1]], axis=1)
        self.image_ = cell_hist

    def BlockNorm_HOG(self,block_size=(2,2)):
        eps=1e-5
        h,w,c,b = self.image_.shape
        h2 = h-block_size[0]+1
        w2 = w-block_size[1]+1
        blockHOG = np.zeros((h2, w2, c, block_size[0], block_size[1], b))
        tmp = np.zeros((h, w2, c, block_size[1],b))
        for i in range(w2):
            for j in range(block_size[1]):
                tmp[:,i,:,j] = self.image_[:,i+j]
        for i in range(h2):
            for j in range(block_size[0]):
                blockHOG[i, :, :, j] = tmp[i+j]
        blockHOG = blockHOG.reshape((h2,w2,c,-1))
        blocknormHOG = blockHOG/(np.sqrt(np.sum(blockHOG**2, axis=3)+eps**2)[:, :, :, None])
        blocknormHOG = blocknormHOG.reshape((h2, w2, c, block_size[0], block_size[1], b))      
        self.image_1 = blocknormHOG

    def _DrawCell(self, bin_idx, bin_len=1, cell_size=(16, 16)):
        h,w = cell_size
        cell = np.zeros((h, w))
        angle = np.pi*((bin_idx*20+90))/180
        h_len=  bin_len*h*np.sin(angle)
        w_len = bin_len*w*np.cos(angle)
        y1 = round((h-h_len)/2)
        y2 = round((h+h_len)/2)
        x1 = round((w+w_len)/2)
        x2 = round((w-w_len)/2)
        cv.line(cell, (x1, y1), (x2, y2), 1)
        return cell

    def main(self,cell_size=16):
        self.Gradients()
        self._HOGinCell()
        self.BlockNorm_HOG()
        h,w,c,b = self.image_1[:, :, :, 0, 0, :].shape
        norm_hist = (255/np.max(self.image_1[:, :, :, 0, 0, :])) * self.image_1[:, :, :, 0, 0, :]
        canvas = None
        canvas = np.zeros((h*cell_size, w*cell_size,c))
        for i in range(h):
            for j in range(w):
                for k in range(b):
                    for l in range(c):
                        canvas[i*cell_size:(i+1)*cell_size, j*cell_size: (j+1)*cell_size, l]+=norm_hist[i, j, l, k]*self._DrawCell(k, norm_hist[i, j, l, k], (cell_size, cell_size))
        canvas = np.clip(canvas,0,255)         
        canvas = canvas.astype(np.uint8)
        if(c==1):
            canvas = np.reshape(canvas,(cell_size*h,cell_size*w))
        self.image_res = canvas
        self.image_res = cv.resize(self.image_res, (600, 600))

#Functoin3;
class Canny():

    def __init__(self, image, threshold_low, threshold_high) -> None:
        self.image = image
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

        self.image_res = image.copy()

    def main(self):
    
        smooth_img = cv.GaussianBlur(self.image, ksize=(5, 5), sigmaX=1, sigmaY=1)
        
        Gx = cv.Sobel(smooth_img, cv.CV_64F, 1, 0, 5)
        Gy = cv.Sobel(smooth_img, cv.CV_64F, 0, 1, 5)
            
        edge_gradient = np.abs(Gx) + np.abs(Gy)
        
        angle = np.arctan2(Gy, Gx) * 180 / np.pi
        
        angle = np.abs(angle)
        angle[angle <= 22.5] = 0
        angle[angle >= 157.5] = 0
        angle[(angle > 22.5) * (angle < 67.5)] = 45
        angle[(angle >= 67.5) * (angle <= 112.5)] = 90
        angle[(angle > 112.5) * (angle <= 157.5)] = 135
        
        keep_mask = np.zeros(smooth_img.shape, np.uint8)
        for y in range(1, edge_gradient.shape[0]-1):
            for x in range(1, edge_gradient.shape[1]-1):
                area_grad_intensity = edge_gradient[y-1:y+2, x-1:x+2]
                area_angle = angle[y-1:y+2, x-1:x+2]
                current_angle = area_angle[1,1]
                current_grad_intensity = area_grad_intensity[1,1]
                
                if current_angle == 0:
                    if current_grad_intensity > max(area_grad_intensity[1,0], area_grad_intensity[1,2]):
                        keep_mask[y,x] = 255
                    else:
                        edge_gradient[y,x] = 0
                elif current_angle == 45:
                    if current_grad_intensity > max(area_grad_intensity[2,0], area_grad_intensity[0,2]):
                        keep_mask[y,x] = 255
                    else:
                        edge_gradient[y,x] = 0
                elif current_angle == 90:
                    if current_grad_intensity > max(area_grad_intensity[0,1], area_grad_intensity[2,1]):
                        keep_mask[y,x] = 255
                    else:
                        edge_gradient[y,x] = 0
                elif current_angle == 135:
                    if current_grad_intensity > max(area_grad_intensity[0,0], area_grad_intensity[2,2]):
                        keep_mask[y,x] = 255
                    else:
                        edge_gradient[y,x] = 0
           
        canny_mask = np.zeros(smooth_img.shape, np.uint8)
        canny_mask[(keep_mask>0) * (edge_gradient>self.threshold_low)] = 255

        min_val = np.min(canny_mask)
        max_val = np.max(canny_mask)
        new_img = (canny_mask - min_val) / (max_val - min_val) # 0-1
        new_img *= 255
        self.image_res = canny_mask

#Function4;
class Hough():
    def __init__(self, image) -> None:
        self.image = image
        self.lines = None
        self.image_res = image.copy()

    def preprocessing(self):
        img_gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(img_gray, 100, 150)
        self.lines = cv.HoughLines(edges, 1, np.pi/180, 110, None, 0, 0)
    def main(self):
        self.preprocessing()
        for i in range(0, len(self.lines)):
            rho = self.lines[i][0][0]
            theta = self.lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a*rho
            y0 = b*rho
            pt1 = (int(x0-1000*b), int(y0+1000*a))
            pt2 = (int(x0+1000*b), int(y0-1000*a))
            cv.line(self.image_res, pt1, pt2, (0,0,255), 1, cv.LINE_AA)
    
#Necessary Functions;
def menu():
    print("================= MENU =================")
    print("                                        ")
    print("Function1: Harris                       ")
    print("Function2: HOG                          ")
    print("Function3: Canny                        ")
    print("Function4: Hough                        ")
    print("Function5: Quit                         ")
    print("                                        ")
    print("========================================")

def check_input_menu():
    
    while True:
        selection = input("Enter your selection: ")
        try:
            selection = int(selection)
        except:
            selection = -1
        
        if selection in [i for i in range(1, 6)]:
            break
    
    return selection

def resize(img):
    return cv.resize(img, (600, 600))

def selection_function(selection, img):

    img = cv.resize(img, (600, 600))

    if selection == 1:
        img1 = img.copy()

        img_harris = Harris(img1, 9, 3, 0.004)
        img_harris.main()
        img_harris.image = resize(img_harris.image)

        cv.imshow("img org", img)
        cv.imshow("img harris", img_harris.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif selection == 2:
        img2 = img.copy()

        img_hog = HOG(img2)
        img_hog.main()
        img_hog.image_res = resize(img_hog.image_res)

        cv.imshow("img org", img)
        cv.imshow("img hog", img_hog.image_res)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif selection == 3:
        img_3 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img3 = img_3.copy()
        img_test = img_3.copy()

        img_canny = Canny(img3, 100, 200)
        img_canny.main()
        img_canny.image_res = resize(img_canny.image_res)
        img_test = cv.Canny(img_test, 100, 200)

        cv.imshow("img org", img)
        cv.imshow("img my canny", img_canny.image_res)
        cv.imshow("img canny", img_test)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif selection == 4:
        img4 = img.copy()

        img_hough = Hough(img4)
        img_hough.main()
        img_hough.image_res = resize(img_hough.image_res)

        cv.imshow("img org", img)
        cv.imshow("img hough", img_hough.image_res)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
    elif selection == 5:
        print("Thank You!")
    else:
        print("Invalid selection!")


def res_your_selection(img_path):

    img = cv.imread(img_path)

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

def main():
    img_path = check_path_img()
    res_your_selection(img_path[1: -1])

#Main;
if __name__ == "__main__":
    main() 







    
