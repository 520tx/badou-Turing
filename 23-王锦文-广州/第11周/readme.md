AlexNet-tfkeras

环境：tensorflow1.15+pthon3.7

该文件夹使用tf自带keras实现alexnet的训练与测试工作，主要实现猫和狗的二分类网络，为加快训练过程，与原始alexnet论文相比，kernel filters减半。

model.py-------构建模型脚本

train.py ---------训练脚本

test.py-------测试脚本

read_dataset.py------加载训练数据

dataset.txt------根据数据集生成的标签文件

训练：python train.py

测试：python test.py

数据集不上传，数据集命名要求：类别.x.jpg,如cat.1.jpg,dog.1.jpg.....

训练后模型考虑到尺寸问题（约25.6M），不上传



resnet50-pytorch

环境：pytorch1.2+python3.7

该文件夹使用pytorch实现resnet50的训练与测试。

model.py---构建resnet模型结构

train.py----训练脚本

test.py---测试脚本

read_dataset.py--数据读取脚本

utils.py----数据增强和预处理脚本

训练：python train.py

测试：python test.py

数据集不上传，数据集命名要求：类别.序号.jpg,如cat.1.jpg,dog.1.jpg.....



