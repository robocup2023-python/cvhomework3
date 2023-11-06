import cv2
import os
import numpy as np

img = cv2.imread("./sourcedata/plane.jpg",0)
print(cv2.IMREAD_GRAYSCALE)

blur = cv2.GaussianBlur(img, (5, 5), 0)

sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

mag = np.sqrt(np.square(sobelx) + np.square(sobely))
mag = np.uint8(mag / np.max(mag) * 255)
theta = np.arctan2(sobely, sobelx) * 180 / np.pi
print(theta[0,0])

nms = np.zeros_like(mag)
for i in range(1, mag.shape[0] - 1):
    if not (i % 100): print(i)
    for j in range(1, mag.shape[1] - 1):
        if np.logical_or(np.logical_and(theta[i, j] >= -22.5, theta[i, j] < 22.5),
                         np.logical_or(theta[i, j] >= 157.5, theta[i, j] < -157.5)):
            if mag[i, j] > mag[i, j - 1] and mag[i, j] > mag[i, j + 1]:
                nms[i, j] = mag[i, j]
        elif np.logical_or(np.logical_and(theta[i, j] >= 22.5, theta[i, j] < 67.5),
                           np.logical_and(theta[i, j] < -112.5, theta[i, j] >= -157.5)):
            if mag[i, j] > mag[i - 1, j + 1] and mag[i, j] > mag[i + 1, j - 1]:
                nms[i, j] = mag[i, j]
        elif np.logical_or(np.logical_and(theta[i, j] >= 67.5, theta[i, j] < 112.5),
                           np.logical_and(theta[i, j] < -67.5, theta[i, j] >= -112.5)):
            if mag[i, j] > mag[i - 1, j] and mag[i, j] > mag[i + 1, j]:
                nms[i, j] = mag[i, j]
        else:
            if mag[i, j] > mag[i - 1, j - 1] and mag[i, j] > mag[i + 1, j + 1]:
                nms[i, j] = mag[i, j]
        # for j in range(1,mag.shape[1]-1):
        #     if (theta[i,j] >= -22.5 and theta[i,j] < 22.5) or (theta[i,j] >= 157.5 and theta[i,j] < -157.5):
        #         if (mag[i,j] > mag[i,j-1] and mag[i,j] > mag[i,j+1]):
        #             nms[i,j] = mag[i,j]
        #     elif (theta[i,j] >= 22.5 and theta[i,j] < 67.5) or (theta[i,j] < -112.5 and theta[i,j] >= -157.5):
        #         #
        #         if mag[i,j] > mag[i-1,j+1] and mag[i,j] > mag[i+1][j-1]:
        #             nms[i,j] = mag[i,j]
        #     elif (theta[i,j] >= 67.5 and theta[i,j] < 112.5) or (theta[i,j] < -67.5 and theta[i,j] >= -112.5):
        #         #
        #         if mag[i,j] > mag[i-1,j] and mag[i,j] > mag[i+1,j]:
        #             nms[i,j] = mag[i,j]
        #     else:
        #         # 左下右上
        #         if mag[i,j] > mag[i-1,j-1] and mag[i,j] > mag[i+1,j+1]:
        #             nms[i,j] = mag[i,j]

# 双阈值算法检测和连接边缘
print(nms.shape)
print(nms[0,0])
print(nms)
low_threshold = 50
high_threshold = 100
edge = np.zeros_like(nms)
strong_i, strong_j = np.where(nms >= high_threshold)
print(strong_i)
weak_i, weak_j = np.where((nms >= low_threshold) & (nms < high_threshold))
edge[strong_i, strong_j] = 1
for i, j in zip(weak_i, weak_j):
    if ((i > 0 and edge[i - 1, j] == 1) or
            (i < edge.shape[0] - 1 and edge[i + 1, j] == 1) or
            (j > 0 and edge[i, j - 1] == 1) or
            (j < edge.shape[1] - 1 and edge[i, j + 1] == 1)):
        edge[i, j] = 1

# 展示结果

if (not os.path.exists("./output/edgedetection")):
    os.makedirs("./output/edgedetection")
cv2.imwrite('./output/edgedetection/Original_Image.jpg', img)
cv2.imwrite('./output/edgedetection/Gaussian_Blur.jpg', blur)
cv2.imwrite('./output/edgedetection/Gradient_Magnitude.jpg', mag)
cv2.imwrite('./output/edgedetection/Non-Maximum_Suppression.jpg', nms)
cv2.imwrite('./output/edgedetection/Canny_Edge_Detection.jpg', edge * 255)
# cv2.imshow('Original Image', img)
# cv2.imshow('Gaussian Blur', blur)
# cv2.imshow('Gradient Magnitude', mag)
# cv2.imshow('Non-Maximum Suppression', nms)
# cv2.imshow('Canny Edge Detection', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
