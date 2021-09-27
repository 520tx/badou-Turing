#coding=utf-8

import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
pca = PCA(n_components=2)   # 降到2维
pca.fit(X)                  # 训练
newX = pca.fit_transform(X)  # 降维后的数据

# PCA(copy=True, n_components=2, whiten=False)
print('降维后的特征数据：\n', newX)  # 输出降维后的数据
print('各主成分的方差值：', pca.explained_variance_)  # 代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分
print('各主成分的方差值占总方差值的比例：', pca.explained_variance_ratio_)  # 降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分
