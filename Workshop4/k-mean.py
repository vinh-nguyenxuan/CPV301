import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class K_mean(object):
    def __init__(self, img, k):
        self.img = img
        self.k = k

    def preprocesisng(self):
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
        vector = np.float32(self.img.reshape((-1,3)))
        vector = vector / 255.0
        return vector
    
    def init_centroids(self, data, k):
        np.random.seed(30)
        centroids = data[np.random.choice(data.shape[0], k, replace=False, ), :]
        return centroids
    
    def calculate_distances(self, data, centroids):
        return np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    
    def cluster_assignment(self, distances):
        return np.argmin(distances, axis=0)
    
    def update_centroids(self, data, k, cluster_ids, centroids):
        for i in range(k):
            centroids[i, :] = np.mean(data[cluster_ids == i, :], axis=0)
        return centroids
    
    def k_means(self, data, k, iter=100):
        centroids = self.init_centroids(data, k)
        for i in range(iter):

            distances = self.calculate_distances(data, centroids)
            
            cluster_ids = self.cluster_assignment(distances)
        
            centroids = self.update_centroids(data, k, cluster_ids, centroids)

        return centroids, cluster_ids
    
    def main(self):
        vector = self.preprocesisng()
        res, labels = self.k_means(vector, self.k)
        img = res[labels]
        img = img.reshape((self.img.shape))
        img = cv.resize(img, (600, 600))
        self.img = cv.resize(self.img, (600, 600))
        plt.figure(figsize=(20, 15))
        plt.subplot(1,2,1)
        plt.imshow(self.img)
        plt.title('Original Image')
        plt.xticks([]), plt.yticks([])
        plt.subplot(1,2,2),plt.imshow(img)
        plt.title('Segmented Image when K = %i' % self.k), plt.xticks([]), plt.yticks([])
        plt.show()

#Test; 
path_img = "C:/Users/votru/Downloads/muvodichc1.jpg"
img_org = cv.imread(path_img)
kmean = K_mean(img_org, 10)
kmean.main()

