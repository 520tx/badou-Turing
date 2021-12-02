# 第四周作业 1.canny

import cv2

img = cv2.imread('lenna.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('canny', cv2.Canny(gray, 200, 300))  # Canny(单通道灰度图,滞后阈值1,滞后阈值2)

cv2.waitKey()
cv2.destroyAllWindows()

# todo 公式
# 1.图像灰度化
# 2.高斯滤波
# 3.图像水平、垂直、对角边缘检测
# 4.梯度幅值进行非极大值抑制
# 5.双阈值算法检测和连接边缘
