import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def stitching(image_paths):
    imgs = []
    for i in range(len(image_paths)):
        img = cv.imread(image_paths[i])
        img = cv.resize(img, (224, 224))
        imgs.append(img)
  
    stitchy=cv.Stitcher.create()
    (dummy,output)=stitchy.stitch(imgs)

    return output


def stitching_build(path0, path1):

    src_img = cv.imread(path0)
    tar_img = cv.imread(path1)
    
    src_gray = cv.cvtColor(src_img, cv.COLOR_RGB2GRAY)
    tar_gray = cv.cvtColor(tar_img, cv.COLOR_RGB2GRAY)

    SIFT_detector = cv.SIFT_create()
    kp1, des1 = SIFT_detector.detectAndCompute(src_gray, None)
    kp2, des2 = SIFT_detector.detectAndCompute(tar_gray, None)

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

    rawMatches = bf.knnMatch(des1, des2, 2)
    matches = []
    ratio = 0.75

    for m,n in rawMatches:
        if m.distance < n.distance * 0.75:
            matches.append(m)

    matches = sorted(matches, key=lambda x: x.distance, reverse=True)
    matches = matches[:200]

    img3 = cv.drawMatches(src_img, kp1, tar_img, kp2, matches, None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    kp1 = np.float32([kp.pt for kp in kp1])
    kp2 = np.float32([kp.pt for kp in kp2])

    pts1 = np.float32([kp1[m.queryIdx] for m in matches])
    pts2 = np.float32([kp2[m.trainIdx] for m in matches])

    (H, status) = cv.findHomography(pts1, pts2, cv.RANSAC)

    h1, w1 = src_img.shape[:2]
    h2, w2 = tar_img.shape[:2]
    result = cv.warpPerspective(src_img, H, (w1+w2, h1))
    result[0:h2, 0:w2] = tar_img
    
    return result

#Test
# imgs = stiting_build("/images/img0.jpg", "/images/img1.jpg")
# plt.imshow(imgs)
# plt.show()
