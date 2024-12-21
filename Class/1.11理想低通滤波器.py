import cv2
import numpy as np
import matplotlib.pyplot as plt


def ideal_low_pass_filter(image, kernel_size):
    # 创建一个均值滤波器（理想低通滤波器）
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    # 对图像应用滤波器（卷积操作）
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image


# 读取图像
image = cv2.imread('my.jpg', cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
if image is None:
    raise ValueError("Image not found!")

# 设置滤波器的大小（影响模糊程度）
kernel_size = 15  # 增加此值可以增加滤波的模糊效果

# 使用理想低通滤波器处理图像
filtered_image = ideal_low_pass_filter(image, kernel_size)

# 显示原图与滤波后的图像
plt.figure(figsize=(12, 6))

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 显示滤波后的图像
plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Ideal Low-Pass Filter)')
plt.axis('off')

plt.tight_layout()
plt.show()
