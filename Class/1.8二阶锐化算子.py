import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image_path = 'my.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 应用拉普拉斯算子进行锐化
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# 将拉普拉斯算子得到的图像转换为8位无符号整数类型
laplacian = np.uint8(np.absolute(laplacian))

# 进行锐化处理
sharpened_image = cv2.addWeighted(image, 1.5, laplacian, -0.5, 0)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sharpened_image, cmap='gray')
plt.title("Sharpened Image with Laplacian")
plt.axis('off')

plt.show()
