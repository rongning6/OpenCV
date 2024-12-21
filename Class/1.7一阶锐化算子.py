import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image_path = 'my.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 应用Sobel算子进行边缘检测
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # 水平边缘
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # 垂直边缘

# 计算梯度幅值
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

# 将结果转换回8位无符号整数类型
sobel_magnitude = np.uint8(np.absolute(sobel_magnitude))

# 进行锐化处理
sharpened_image = cv2.addWeighted(image, 1.5, sobel_magnitude, -0.5, 0)

# 显示原始图像与锐化后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sharpened_image, cmap='gray')
plt.title("Sharpened Image with Sobel")
plt.axis('off')

plt.show()
