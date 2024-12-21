import numpy as np
import cv2

def gaussian_kernel(kernel_size, sigma):
    # 计算高斯核的半径
    pad = kernel_size // 2
    # 创建一个空的高斯核
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    # 高斯核的中心坐标
    center = kernel_size // 2

    # 计算高斯核的每个值
    for i in range(kernel_size):
        for j in range(kernel_size):
            # 计算每个位置的高斯函数值
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # 归一化高斯核
    kernel /= np.sum(kernel)

    return kernel

def gaussian_filter(image, kernel_size, sigma):
    # 获取图像的高度和宽度
    height, width, channels = image.shape

    # 创建输出图像
    output_image = np.zeros_like(image)

    # 获取高斯核
    kernel = gaussian_kernel(kernel_size, sigma)

    # 计算滤波器的半径
    pad = kernel_size // 2

    # 对图像进行边界填充（0填充）
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    # 遍历每个像素进行高斯滤波
    for i in range(height):
        for j in range(width):
            # 提取当前区域
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            # 使用高斯核对区域进行卷积，计算加权平均
            for c in range(channels):
                output_image[i, j, c] = np.sum(region[:, :, c] * kernel)

    return output_image

# 读取图像
image = cv2.imread('my.jpg')

# 使用高斯滤波器
filtered_image = gaussian_filter(image, 5, 1.0)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)

# 等待按键并关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
