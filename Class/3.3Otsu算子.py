import cv2
import matplotlib.pyplot as plt

def otsu_thresholding(image_path):
    # 1. 读取图像并转换为灰度图
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 使用高斯滤波器降噪（可选，但推荐）
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 应用Otsu方法进行全局阈值分割
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 显示原图像、灰度图像和分割结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1), plt.title("Original Image"), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2), plt.title("Grayscale Image"), plt.imshow(gray, cmap='gray')
    plt.subplot(1, 3, 3), plt.title("Otsu Thresholding"), plt.imshow(otsu_thresh, cmap='gray')
    plt.show()

    return otsu_thresh

# 使用示例
otsu_image = otsu_thresholding("my.jpg")