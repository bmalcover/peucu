import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt


img1 = cv2.imread("data2/p_0001/color/000.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("data2/p_0001/color/001.png", cv2.IMREAD_GRAYSCALE)

img1d = cv2.imread("data2/p_0001/depth/000.png", cv2.IMREAD_UNCHANGED)
img2d = cv2.imread("data2/p_0001/depth/001.png", cv2.IMREAD_UNCHANGED)



FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6,  # 12
                key_size=12,     # 20
                multi_probe_level=1)  #2


# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]
# ratio test as per Lowe's paper

for i, (m, n) in enumerate(matches):
    if m.distance < 0.4 * n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv2.DrawMatchesFlags_DEFAULT)
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
plt.imshow(img3,)
plt.show()
