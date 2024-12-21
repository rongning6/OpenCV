import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('my.jpg', cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
if image is None:
    raise ValueError("Image not found!")

# 对图像进行傅里叶变换（DFT）
dft = np.fft.fft2(image)  # 2D傅里叶变换
dft_shift = np.fft.fftshift(dft)  # 将低频成分移到图像的中心

# 计算频谱的幅度（模值）
magnitude_spectrum = np.abs(dft_shift)

# 显示原始图像和频谱图像
plt.figure(figsize=(12, 6))

# 显示原图
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 显示频谱（幅度图）
plt.subplot(1, 2, 2)
plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')  # 使用log进行显示增强
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.tight_layout()
plt.show()

