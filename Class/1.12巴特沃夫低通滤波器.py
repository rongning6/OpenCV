import cv2
import numpy as np
import matplotlib.pyplot as plt


def butterworth_lowpass_filter(shape, cutoff, order=2):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.ogrid[:rows, :cols]
    d = np.sqrt((x - crow) ** 2 + (y - ccol) ** 2)
    h = 1 / (1 + (d / cutoff) ** (2 * order))
    return h


def apply_butterworth_filter(image, cutoff, order=2):
    # 对图像进行傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # 将零频率移动到频域中心

    # 获取巴特沃夫低通滤波器
    filter_kernel = butterworth_lowpass_filter(image.shape, cutoff, order)

    # 应用滤波器
    fshift_filtered = fshift * filter_kernel

    # 对滤波后的频域图像进行逆傅里叶变换
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))

    return filtered_image


# 读取图像
image = cv2.imread('my.jpg', cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
if image is None:
    raise ValueError("Image not found!")

# 设置巴特沃夫低通滤波器的参数
cutoff = 50  # 截止频率
order = 2  # 阶数

# 使用巴特沃夫低通滤波器处理图像
filtered_image = apply_butterworth_filter(image, cutoff, order)

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
plt.title('Filtered Image (Butterworth Low-Pass Filter)')
plt.axis('off')

plt.tight_layout()
plt.show()


