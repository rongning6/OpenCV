import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_lowpass_filter(shape, cutoff):
    """
    创建高斯低通滤波器
    shape: 图像的尺寸 (高度, 宽度)
    cutoff: 截止频率（影响滤波的模糊程度）
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2  # 图像中心坐标

    # 创建坐标网格
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)

    # 计算频率距离
    D = np.sqrt(X**2 + Y**2)

    # 计算高斯低通滤波器
    H = np.exp(-(D**2) / (2 * (cutoff**2)))

    return H

def apply_gaussian_filter(image, cutoff):
    """
    对图像应用高斯低通滤波器
    image: 输入图像
    cutoff: 截止频率
    """
    # 对图像进行傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # 将零频率移动到频域中心

    # 获取高斯低通滤波器
    filter_kernel = gaussian_lowpass_filter(image.shape, cutoff)

    # 应用滤波器
    fshift_filtered = fshift * filter_kernel

    # 对滤波后的频域图像进行逆傅里叶变换
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))

    return filtered_image

# 读取图像
image = cv2.imread('my.jpg', cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
if image is None:
    raise ValueError("Image not found!")

# 设置高斯低通滤波器的参数
cutoff = 50  # 截止频率，控制模糊程度

# 使用高斯低通滤波器处理图像
filtered_image = apply_gaussian_filter(image, cutoff)

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
plt.title('Filtered Image (Gaussian Low-Pass Filter)')
plt.axis('off')

plt.tight_layout()
plt.show()
