import cv2
import numpy as np
import matplotlib.pyplot as plt


def prewitt_edge_detection(image_path):
    # 1. 读取图像并转为灰度图
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 定义Prewitt算子的卷积核
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])  # 水平方向
    kernel_y = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]])  # 垂直方向

    # 3. 使用卷积核进行边缘检测
    gradient_x = cv2.filter2D(gray, -1, kernel_x)  # 水平方向梯度
    gradient_y = cv2.filter2D(gray, -1, kernel_y)  # 垂直方向梯度

    # 4. 合成梯度幅值
    gradient_magnitude = cv2.magnitude(gradient_x.astype(float), gradient_y.astype(float))

    # 5. 归一化梯度图像到0-255
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    gradient_magnitude = gradient_magnitude.astype(np.uint8)

    # 6. 显示结果
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1), plt.title("Original Image"), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2), plt.title("Gradient X"), plt.imshow(gradient_x, cmap='gray')
    plt.subplot(1, 3, 3), plt.title("Gradient Y"), plt.imshow(gradient_y, cmap='gray')
    plt.figure(), plt.title("Edge Detection (Prewitt)"), plt.imshow(gradient_magnitude, cmap='gray')
    plt.show()

    return gradient_magnitude


# 使用示例
edge_image = prewitt_edge_detection("my.jpg")
