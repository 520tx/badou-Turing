import cv2
import numpy as np

# [1] 加载图像
img = cv2.imread('photo.jpg')
cv2.imshow("src", img)
result1 = img.copy()
print(img.shape)

# [2] 确定变换前后4个相应点的坐标
# 注意: 这里src和dst的输入并不是图像，而是图像对应的顶点坐标
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

# [3] 生成 '透视变换矩阵m'
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)

# [4] 进行透视变换
result = cv2.warpPerspective(result1, m, (337, 488))
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
