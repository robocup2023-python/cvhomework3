import cv2
import os
import numpy as np


def harris_corner_detection(image, ksize=3, k=0.04, threshold=0.01):
    image_smoothed = cv2.GaussianBlur(image, (ksize, ksize), 0)
    print(image_smoothed.shape)

    dx = cv2.Sobel(image_smoothed, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image_smoothed, cv2.CV_64F, 0, 1, ksize=3)

    # cv2.imshow("test.jpg", dx)
    cv2.waitKey(0)
    if not os.path.exists("./output"):
        os.mkdir("./output")

    cv2.imwrite("output/cornerdetection/cornerdetection/dx.jpg", dx)
    cv2.imwrite("./output/cornerdetection/dy.jpg", dy)
    cv2.imwrite("./output/cornerdetection/dx.jpg", dx)
    dxx = dx * dx
    dyy = dy * dy
    dxy = dx * dy
    print(f"dx shape: {dx.shape},dxx shape: {dxx.shape}")

    dxx_smoothed = cv2.GaussianBlur(dxx, (ksize, ksize), 1.5)
    dyy_smoothed = cv2.GaussianBlur(dyy, (ksize, ksize), 1.5)
    dxy_smoothed = cv2.GaussianBlur(dxy, (ksize, ksize), 1.5)

    M = np.zeros((image.shape[0], image.shape[1], 2, 2))
    print(M.shape)
    print(dxx_smoothed.shape)
    M[..., 0, 0] = dxx_smoothed
    M[..., 0, 1] = dxy_smoothed
    M[..., 1, 0] = dxy_smoothed
    M[..., 1, 1] = dyy_smoothed

    det_M = np.linalg.det(M)
    trace_M = np.trace(M, axis1=2, axis2=3)
    R = det_M - k * trace_M ** 2

    corner_candidates = np.zeros_like(image)
    corner_candidates[R > threshold * R.max()] = 255

    # 获取边角点
    corners = cv2.cornerHarris(corner_candidates.astype(np.float32), blockSize=2, ksize=3,k=k)
    # 膨胀
    corners = cv2.dilate(corners, None)
    print("Corners shape is: ", corners.shape)
    corner_points = np.argwhere(corners > 0.01 * corners.max())

    return corner_points


if __name__ == "__main__":
    image = cv2.imread("./sourcedata/plane.jpg",0)
    corners = harris_corner_detection(image)
    for corner in corners:
        x, y = corner
        cv2.circle(image, (y, x), 1, 255, 0)

    cv2.imshow("Harris Corner Detection", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("./output/cornerdetection/harriscorner1.jpg",image)
