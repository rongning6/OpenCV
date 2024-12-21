import cv2
import matplotlib.pyplot as plt

def canny_edge_detection(image_path, low_threshold=50, high_threshold=150):
    # 1. 读取图像并转换为灰度图
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 应用高斯滤波器进行降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # 3. 使用Canny算子进行边缘检测
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # 4. 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.title("Original Image"), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2), plt.title("Canny Edges"), plt.imshow(edges, cmap='gray')
    plt.show()

    return edges

# 使用示例
edge_image = canny_edge_detection("my.jpg")