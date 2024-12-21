from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = 'my.jpg'  # 替换为你的图片路径
img = Image.open(image_path).convert('L')  # 将图像转换为灰度模式

# 将图像转换为 NumPy 数组
img_array = np.array(img)

# 设置切片的灰度级别范围
# 例如，将灰度值分成10个区间
num_slices = 4
slices = np.linspace(0, 255, num_slices + 1, dtype=int)

# 创建一个空的列表来保存切片图像
slice_images = []

# 按灰度级进行切片
for i in range(num_slices):
    # 创建掩码，选择在当前切片范围内的像素
    mask = (img_array >= slices[i]) & (img_array < slices[i + 1])
    # 将掩码应用到原图像
    sliced_img = np.zeros_like(img_array)
    sliced_img[mask] = img_array[mask]

    # 将切片图像转换为PIL对象并添加到列表
    slice_images.append(Image.fromarray(sliced_img))

# 显示所有切片
fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))
for i, slice_img in enumerate(slice_images):
    axes[i].imshow(slice_img, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'Slice {i + 1}')

plt.tight_layout()
plt.show()
