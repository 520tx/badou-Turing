import cv2

if __name__ == '__main__':
    # [1] 加载图像：1(默认参数)代表加载彩色图片
    img = cv2.imread("lenna.png", 1)
    # [2] 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # [3] Canny边缘检测：低滞后阈值为200，高滞后阈值为300
    dst = cv2.Canny(gray, 200, 300)
    # [4] 可视化
    cv2.imshow("canny", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
