import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_hog_features(image_path, block_size=(2, 2), cell_size=(8, 8), nbins=9):
    # 1. 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not found!")
        return None

    # 2. 创建 HOG 描述符对象
    hog = cv2.HOGDescriptor(
        _winSize=(image.shape[1] // 8 * 8, image.shape[0] // 8 * 8),  # 设置窗口大小为图像大小的倍数
        _blockSize=(block_size[0] * cell_size[0], block_size[1] * cell_size[1]),  # 块的大小
        _blockStride=(cell_size[0], cell_size[1]),  # 块的步长
        _cellSize=(cell_size[0], cell_size[1]),  # 每个单元的大小
        _nbins=nbins  # 梯度方向的数量
    )

    # 3. 计算图像的 HOG 特征
    hog_features = hog.compute(image)

    # 4. 将 HOG 特征转化为一维向量
    hog_features = hog_features.flatten()

    # 5. 返回 HOG 特征
    return hog_features


def visualize_hog(image_path):
    # 1. 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not found!")
        return

    # 2. 创建 HOG 描述符对象
    hog = cv2.HOGDescriptor()

    # 3. 计算图像的 HOG 特征
    hog_features = hog.compute(image)

    # 4. 绘制 HOG 特征的方向直方图
    # 为了演示，我们只取一个小的 cell 来可视化其方向分布
    num_cells = hog_features.shape[0]
    cell_width = 8
    cell_height = 8
    n_bins = 9  # 梯度方向数量

    # 创建方向直方图
    hist = np.zeros((n_bins, 1))
    for i in range(num_cells):
        hist[i % n_bins] = hog_features[i]

    # 显示图像及其 HOG 特征
    plt.figure(figsize=(10, 6))

    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # 方向直方图
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(n_bins), hist.flatten(), color='gray')
    plt.title("HOG Feature Histogram")
    plt.xlabel("Gradient Direction")
    plt.ylabel("Magnitude")
    plt.show()


# 示例：提取 HOG 特征并可视化
image_path = 'my.jpg'  # 替换为你的图像路径
hog_features = extract_hog_features(image_path)
print("Extracted HOG Features Length:", len(hog_features))

# 可视化原始图像及其 HOG 特征
visualize_hog(image_path)
