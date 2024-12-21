import cv2
import numpy as np
import matplotlib.pyplot as plt


# 生成高斯噪声的函数
def add_gaussian_noise(image, mean=0, std_dev=25):
    rows, cols = image.shape
    # 生成符合正态分布的噪声
    noise = np.random.normal(mean, std_dev, (rows, cols))
    # 将噪声加到原图像上
    noisy_image = image + noise
    # 限制图像像素值在0到255之间
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image


# 使用均值滤波器去噪的函数
def remove_noise_using_mean_filter(noisy_image, kernel_size=5):
    # 应用均值滤波器
    denoised_image = cv2.blur(noisy_image, (kernel_size, kernel_size))
    return denoised_image


# 显示图像的辅助函数
def display_images(original, noisy, denoised):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(denoised, cmap='gray')
    plt.title('Denoised Image (Mean Filter)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 主程序
def main(image_path):
    # 读取原始图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return

    # 添加高斯噪声
    noisy_image = add_gaussian_noise(image, mean=0, std_dev=25)

    # 使用均值滤波器去噪
    denoised_image = remove_noise_using_mean_filter(noisy_image, kernel_size=5)

    # 显示原图、噪声图和去噪后的图像
    display_images(image, noisy_image, denoised_image)


# 调用主程序，输入图像路径
main("my.jpg")

