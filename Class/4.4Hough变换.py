import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('my.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Canny边缘检测
edges = cv2.Canny(gray, 50, 150)

# 使用概率Hough变换检测直线
# 1. edges：边缘检测的图像
# 2. 1：距离分辨率（一般为1，表示每个像素为单位）
# 3. np.pi / 180：角度分辨率，每次旋转1度
# 4. 100：累加器的阈值，投票数大于此值则认为是直线
# 5. minLineLength：直线的最小长度，小于此值的线段会被排除
# 6. maxLineGap：两个线段的最大间隔，两个相距较近的线段会被认为是一个直线
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

# 在图像上绘制检测到的直线
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示结果
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Improved Hough Line Transform")
plt.show()
