# 第四周作业 3.透视变换
import cv2
import numpy as np


# 矩阵变换公式实现
def getWarpMatrix(src, dst):
    """
    src:数组>=4&==dst
    dst:数组>=4&==src
    """
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4, '选择区域坐标点数组小于4'

    dim = src.shape[0]
    A = np.zeros((2 * dim, 8))
    B = np.zeros((2 * dim, 1))
    for i in range(0, dim):
        A_i = src[i, :]  # A矩阵i行
        B_i = dst[i, :]  # B矩阵i行
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]
    A = np.mat(A)
    warp_matrix = A.I * B  # A.I是求逆矩阵
    warp_matrix = np.array(warp_matrix).T[0]
    warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1.0, axis=0)
    warp_matrix = warp_matrix.reshape(3, 3)
    return warp_matrix


if __name__ == "__main__":
    img = cv2.imread("1.jpg")
    imgcopy = img.copy()
    src = np.float32([[0, 0], [0, 512], [512, 0], [512, 512]])  # 原图坐标
    dst = np.float32([[100, 100], [100, 612], [412, 0], [412, 412]])  # 变换坐标
    pers = cv2.getPerspectiveTransform(src, dst)  # 计算变换矩阵
    #pers = getWarpMatrix(src, dst)
    imgwarp = cv2.warpPerspective(imgcopy, pers, (612, 612))  # 变换图片
    cv2.imshow("img", img)
    cv2.imshow("imgwarp", imgwarp)
    cv2.waitKey(0)
