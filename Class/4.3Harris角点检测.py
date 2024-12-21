import cv2
import numpy as np
import matplotlib.pyplot as plt


def harris_corner_detection(image_path, block_size=3, ksize=3, k=0.04, threshold=0.01):
    # 1. 读取图像并转换为灰度图
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 计算图像的梯度
    gray_image = np.float32(gray_image)
    # 计算 x 和 y 方向的梯度
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)

    # 3. 计算 Harris 响应矩阵 M
    grad_x2 = grad_x ** 2
    grad_y2 = grad_y ** 2
    grad_xy = grad_x * grad_y

    # 使用高斯加权计算每个像素的结构张量
    Mxx = cv2.GaussianBlur(grad_x2, (block_size, block_size), 0)
    Myy = cv2.GaussianBlur(grad_y2, (block_size, block_size), 0)
    Mxy = cv2.GaussianBlur(grad_xy, (block_size, block_size), 0)

    # 4. 计算 Harris 响应函数 R
    det_M = Mxx * Myy - Mxy ** 2
    trace_M = Mxx + Myy
    R = det_M - k * (trace_M ** 2)

    # 5. 进行非极大值抑制
    corner_responses = np.zeros_like(R)
    corner_responses[R > threshold * R.max()] = 255

    # 6. 在图像上标记角点
    corner_responses = np.uint8(corner_responses)
    corners = cv2.dilate(corner_responses, None)

    # 7. 绘制角点
    result_image = image.copy()
    result_image[corners == 255] = [0, 0, 255]  # 标记角点为红色

    return result_image, corners


# 主程序
image_path = "my.jpg"  # 替换为你的图像路径
result_image, corners = harris_corner_detection(image_path)

# 显示结果
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title("Harris Corner Detection")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(corners, cmap='gray')
plt.title("Corner Detection Result")
plt.axis('off')

plt.show()
