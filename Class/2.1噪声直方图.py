import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_gaussian_noise(image, mean=0, sigma=25):

    # 获取图像的行列数
    row, col = image.shape

    # 生成高斯噪声
    gaussian_noise = np.random.normal(mean, sigma, (row, col))

    # 添加噪声到原图像
    noisy_image = np.clip(image + gaussian_noise, 0, 255)  # 保证像素值在0到255之间
    return noisy_image.astype(np.uint8)


def plot_noise_histogram(noisy_image):

    # 计算图像的直方图
    hist, bins = np.histogram(noisy_image.flatten(), bins=256, range=(0, 255))

    # 绘制直方图
    plt.figure(figsize=(8, 6))
    plt.title("Noise Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.bar(bins[:-1], hist, width=1, color='gray', alpha=0.7)
    plt.grid(True)
    plt.show()


# 读取原图像
image = cv2.imread('my.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found!")

# 向图像添加高斯噪声
noisy_image = add_gaussian_noise(image, mean=0, sigma=25)

# 显示原图像和噪声图像
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# 绘制噪声直方图
plot_noise_histogram(noisy_image)
