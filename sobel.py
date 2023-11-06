import cv2
import numpy as np

image = cv2.imread('./sourcedata/plane.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel算子
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
output_image = np.zeros_like(image)

# 边缘填充,这里使用零填充
padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

# 进行滤波操作
for i in range(1, padded_image.shape[0] - 1):
    for j in range(1, padded_image.shape[1] - 1):
        neighborhood = padded_image[i - 1:i + 2, j - 1:j + 2]
        gradient_x = np.sum(sobel_x * neighborhood)
        gradient_y = np.sum(sobel_y * neighborhood)
        output_image[i - 1, j - 1] = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

# 幅值归一化
output_image = cv2.normalize(output_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 输出滤波后的图像
cv2.imwrite('./output/sobelfilter/test.jpg', output_image)
