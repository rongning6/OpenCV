import numpy as np
import cv2
import matplotlib.pyplot as plt


def rgb_to_his(image):
    # 将RGB图像归一化到0-1之间
    image = image.astype(np.float32) / 255.0

    # 提取RGB通道
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # 计算色调(Hue)
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    H = np.arctan2(num, den) * (180 / np.pi)  # 色调范围 [-180, 180]

    # 计算饱和度(Saturation)
    Cmax = np.maximum(R, np.maximum(G, B))
    Cmin = np.minimum(R, np.minimum(G, B))
    S = (Cmax - Cmin) / (Cmax + 1e-6)

    # 计算亮度(Intensity)
    I = (Cmax + Cmin) / 2

    # 将色调值范围从[-180, 180]调整到[0, 360]，以便显示
    H = (H + 180) % 360

    return H, S, I


# 读取RGB图像
image = cv2.imread('my.jpg')  # 替换为你的图片路径

# 转换为HIS色彩空间
H, S, I = rgb_to_his(image)

# 显示HIS各个通道
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(H, cmap='hsv')
plt.title("Hue (H)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(S, cmap='gray')
plt.title("Saturation (S)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(I, cmap='gray')
plt.title("Intensity (I)")
plt.axis('off')

plt.show()

def his_to_rgb(H, S, I):
    # 将H, S, I转回RGB
    H = H / 360.0  # 色调H范围 [0, 1]
    S = np.clip(S, 0, 1)  # 饱和度S范围 [0, 1]
    I = np.clip(I, 0, 1)  # 亮度I范围 [0, 1]

    # 计算RGB色彩空间
    C = (1 - np.abs(2 * I - 1)) * S  # Chroma
    X = C * (1 - np.abs((H * 6) % 2 - 1))  # Intermediate value
    m = I - C / 2  # Match value

    # 获取RGB的三个分量
    H_prime = H * 6
    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)

    # 根据H的范围计算RGB
    mask = (H_prime >= 0) & (H_prime < 1)
    R[mask], G[mask], B[mask] = C, X, 0
    mask = (H_prime >= 1) & (H_prime < 2)
    R[mask], G[mask], B[mask] = X, C, 0
    mask = (H_prime >= 2) & (H_prime < 3)
    R[mask], G[mask], B[mask] = 0, C, X
    mask = (H_prime >= 3) & (H_prime < 4)
    R[mask], G[mask], B[mask] = 0, X, C
    mask = (H_prime >= 4) & (H_prime < 5)
    R[mask], G[mask], B[mask] = X, 0, C
    mask = (H_prime >= 5) & (H_prime < 6)
    R[mask], G[mask], B[mask] = C, 0, X

    # 将RGB值映射回0-255并添加m
    R = (R + m) * 255
    G = (G + m) * 255
    B = (B + m) * 255

    # 返回RGB图像
    return np.stack([R, G, B], axis=-1).astype(np.uint8)


# 假设H, S, I已经计算出来（来自上面的rgb_to_his函数）
# 这里我们使用一个假设的值来进行转换
H, S, I = np.random.rand(256, 256) * 360, np.random.rand(256, 256), np.random.rand(256, 256)

# 将HIS图像转换回RGB
rgb_image = his_to_rgb(H, S, I)

# 显示结果
plt.imshow(rgb_image)
plt.title("Reconstructed RGB Image")
plt.axis('off')
plt.show()

