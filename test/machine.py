import numpy as np
import cv2

def harris_corner_detection(image, ksize=3, k=0.04, threshold=0.01):
    image_smoothed = cv2.GaussianBlur(image, (ksize, ksize), 0)
    print(image_smoothed.shape)

    # 2. 计算梯度
    dx = cv2.Sobel(image_smoothed, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image_smoothed, cv2.CV_64F, 0, 1, ksize=3)
    print(dx.shape)

    # 3. 计算梯度乘积、梯度平方和梯度叉积
    dxx = dx * dx
    dyy = dy * dy
    dxy = dx * dy
    print(dxx.shape)

    # 4. 使用高斯滤波平滑中间结果
    dxx_smoothed = cv2.GaussianBlur(dxx, (ksize, ksize), 0)
    dyy_smoothed = cv2.GaussianBlur(dyy, (ksize, ksize), 0)
    dxy_smoothed = cv2.GaussianBlur(dxy, (ksize, ksize), 0)

    # 5. 构造二阶矩阵M
    M = np.zeros((image.shape[0], image.shape[1], 2, 2))
    M[..., 0, 0] = dxx_smoothed
    M[..., 0, 1] = dxy_smoothed
    M[..., 1, 0] = dxy_smoothed
    M[..., 1, 1] = dyy_smoothed

    # 6. 计算角点响应函数R
    det_M = np.linalg.det(M)
    trace_M = np.trace(M, axis1=2, axis2=3)
    R = det_M - k * (trace_M ** 2)

    # 7. 根据阈值筛选候选角点
    corner_candidates = np.zeros_like(image)
    corner_candidates[R > threshold * R.max()] = 255

    # 8. 非最大化抑制
    corners = cv2.cornerHarris(corner_candidates.astype(np.float32), blockSize=2, ksize=3,k=0.04)
    corners = cv2.dilate(corners, None)
    corner_points = np.argwhere(corners > 0.01 * corners.max())

    return corner_points

# 读取图像
image = cv2.imread('../sourcedata/seaisland.jpg', 0)

# 执行角点检测
corners = harris_corner_detection(image)

# 在图像上绘制角点
for corner in corners:
    x, y = corner
    cv2.circle(image, (y, x), 3, 255, -1)

# 显示结果
cv2.imshow('Harris Corner Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()