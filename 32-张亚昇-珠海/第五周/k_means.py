import numpy as np
import random
data = np.random.rand(10, 2)
def k_means(data, n_cluster):
    #m = len(data)
    k_center_size = np.random.randint(0, data.shape[0]-1, [1, n_cluster])
    distance = 0
    dis_matrix = np.zeros((n_cluster, data.shape[0]))
    for i in range(n_cluster):
        for j in range(data.shape[0]):
            distance = 0
            for k in range(data.shape[1]):
                distance += (data[k_center_size[0, i:i+1]][0, k] - data[j][k])**2
            dis_all = np.sqrt(distance)
            dis_matrix[i, j] = dis_all


    #k_center = random.sample(data, n_cluster)
    #for i in range()
    return dis_matrix

kk = k_means(data, 3)
print(kk)