import pandas as pd


sales = pd.read_csv('train_data.csv',sep = '\s*,\s*',engine='python')
X = sales['X'].values
Y = sales['Y'].values

s1 = 0
s2 = 0
s3 = 0
s4 = 0
n = 4
for i in range(n):
    s1 = s1 +X[i]*Y[i]
    s2 = s2 +X[i]
    s3 = s3 +Y[i]
    s4 = s4 +X[i]*X[i]
k = (s2*s3-n*s1)/(s2*s2-s4*n)
b = (s3-k*s2)/n
print('Coeff:{}Intercept:{}'.format(k,b))