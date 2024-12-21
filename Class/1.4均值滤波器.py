import numpy as np
import cv2

def mean_filter(image, kernel_size):
    # 获取图像的高度和宽度
    height, width, channels = image.shape
    # 创建输出图像
    output_image = np.zeros_like(image)

    # 计算滤波器的半径
    pad = kernel_size // 2

    # 对图像进行边界填充（0填充）
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    # 遍历每个像素进行均值滤波
    for i in range(height):
        for j in range(width):
            # 提取当前区域
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            # 计算区域的均值
            output_image[i, j] = np.mean(region, axis=(0, 1))

    return output_image


# 读取图像
image = cv2.imread('my.jpg')

# 使用自定义均值滤波器
filtered_image = mean_filter(image, 5)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)

# 等待按键并关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
