from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image_path = 'my.jpg'  # 替换为你的图片路径
img = Image.open(image_path).convert('L')  # 转换为灰度模式

# 将图像转换为 NumPy 数组
img_array = np.array(img)

# 计算直方图
histogram, bin_edges = np.histogram(img_array, bins=256, range=(0, 255))

# 显示直方图
plt.figure(figsize=(8, 6))
plt.title('Gray Histogram')
plt.xlabel('Pixel Value (0-255)')
plt.ylabel('Frequency')
plt.bar(bin_edges[:-1], histogram, width=1, color='gray', alpha=0.7)
plt.xlim(0, 255)  # 设置X轴的范围
plt.grid(True)
plt.show()

# 打印直方图的统计信息
print("Histogram Statistics:")
print(f"Total Pixels: {np.sum(histogram)}")
print(f"Min Pixel Value: {np.min(img_array)}")
print(f"Max Pixel Value: {np.max(img_array)}")
print(f"Mean Pixel Value: {np.mean(img_array):.2f}")
print(f"Standard Deviation: {np.std(img_array):.2f}")
