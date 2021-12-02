import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pylab as pl
from scipy.optimize import leastsq

# 数据量。
SIZE = 50
# 创建等差数列，产生数据。np.linspace 返回一个一维数组，SIZE指定数组长度。
# 数组最小值是0，最大值是10。所有元素间隔相等。
X = np.linspace(0, 10, SIZE)
Y = 3 * X + 10

n = 2
# 目标函数
def real_func(x):
    return 3 * x + 10


# 多项式函数
def fit_func(p, x):
    f = np.poly1d(p)  # 生成多项式函数 np.poly1d([2,3,5,7]) =>f(x) = 2x^3+3x^2+5x^1+7
    return f(x)


# 残差函数
def residuals_func(p, y, x):
    ret = fit_func(p, x) - y
    return ret

# 让散点图的数据更加随机并且添加一些噪声。
random_x = []
random_y = []
# 添加直线随机噪声
for i in range(25):
    random_x.append(X[i] + random.uniform(-0.5, 0.5))
    random_y.append(Y[i] + random.uniform(-0.5, 0.5))
# 添加随机噪声
for i in range(25):
    random_x.append(random.uniform(0,10))
    random_y.append(random.uniform(10,40))
RANDOM_X = np.array(random_x) # 散点图的横轴。
RANDOM_Y = np.array(random_y) # 散点图的纵轴

# 使用RANSAC算法估算模型
# 迭代最大次数，每次得到更好的估计会优化iters的数值
iters = 3000
# 数据和模型之间可接受的差值
sigma = 0.25
# 最好模型的参数估计和内点数目
best_a = 0
best_b = 0
pretotal = 0
# 希望的得到正确模型的概率
P = 0.99
for i in range(iters):
    # 随机在数据中红选出两个点去求解模型
    sample_index = random.sample(range(SIZE),2)
    x_1 = RANDOM_X[sample_index[0]]
    x_2 = RANDOM_X[sample_index[1]]
    y_1 = RANDOM_Y[sample_index[0]]
    y_2 = RANDOM_Y[sample_index[1]]

    # y = ax + b 求解出a，b
    a = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - a * x_1

    # 算出内点数目
    total_inlier = 0
    for index in range(SIZE ):
        y_estimate = a * RANDOM_X[index] + b
        if abs(y_estimate - RANDOM_Y[index]) < sigma:
            total_inlier = total_inlier + 1

    # 判断当前的模型是否比之前估算的模型好
    if total_inlier > pretotal:
        iters = math.log(1 - P) / math.log(1 - pow(total_inlier / (SIZE * 2), 2)) #迭代次数推导
        print("iters",iters)
        pretotal = total_inlier
        best_a = a
        best_b = b

    # 判断是否当前模型已经符合超过一半的点
    if total_inlier > SIZE/2:
        print("i",i)
        break

# 用RANSAC得到的最佳估计画图
Y = best_a * RANDOM_X + best_b

x_points = np.linspace(0, 10, 1000)  # 画图时需要的连续点，连续点

# y0 = Y
# y1 = RANDOM_Y
p_init = np.random.randn(n)  # 随机初始化多项式参数

plsq = leastsq(residuals_func, p_init, args=(RANDOM_Y, X)) #最小二乘法拟合函数

print('Fitting Parameters: ', plsq[0])  # 输出拟合参数

pl.figure(figsize=(9,9))#设置子图大小
pl.plot(x_points, real_func(x_points), label='real')
pl.plot(x_points, fit_func(plsq[0], x_points), label='fitted curve')
pl.plot(X, RANDOM_Y, 'bo', label='with noise')

pl.plot(RANDOM_X, Y, label='RANSAC')
pl.legend()
pl.show()

