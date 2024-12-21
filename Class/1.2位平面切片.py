from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image_path = 'my.jpg'  # 替换为你的图片路径
img = Image.open(image_path).convert('L')  # 转换为灰度模式

# 将图像转换为 NumPy 数组
img_array = np.array(img)

# 获取图像的尺寸
height, width = img_array.shape

# 设置位平面的数量（通常为8，因为灰度图像为8位）
num_planes = 8

# 创建一个空的列表来保存每个位平面图像
bit_planes = []

# 生成每个位平面
for i in range(num_planes):
    # 获取当前位的掩码 (1 << i) 提取对应的位
    bit_plane = (img_array & (1 << (num_planes - 1 - i))) >> (num_planes - 1 - i)

    # 将其转换为0或255，便于显示
    bit_plane_img = bit_plane * 255  # 将 0 或 1 转换为 0 或 255

    # 将切片图像添加到列表中
    bit_planes.append(bit_plane_img)

# 显示所有位平面
fig, axes = plt.subplots(2, 4, figsize=(12, 6))  # 2行4列展示8个平面
axes = axes.flatten()

for i in range(num_planes):
    axes[i].imshow(bit_planes[i], cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'Bit Plane {i + 1}')

plt.tight_layout()
plt.show()
