mobilenetv1-pytorch

环境：pytorch1.2+pthon3.7

使用pytorch实现mobilenetv1训练和测试

mobilnetv1.py-------构建模型脚本

train.py ---------训练脚本

test.py-------测试脚本

read_dataset.py------加载训练数据

训练：python train.py --lr 0.01 --epochs 20 --batch_size 32  --savepath ./

测试：python test.py --imgpath ./test/dog1.jpg --load_model ./model_last.pth

数据集不上传(主要是教学上提供的猫和狗25000张图片)，数据集命名要求：类别.x.jpg,如cat.1.jpg,dog.1.jpg.....

训练后模型考虑到尺寸，不上传

inceptionv3-pytorch

环境：pytorch1.2+python3.7

该文件夹使用pytorch实现inceptionv3的训练与测试。

model.py---构建inceptionv3模型结构

train.py----训练脚本

test.py---测试脚本

read_dataset.py--数据读取脚本

utils.py----数据增强和预处理脚本

训练：python train.py

测试：python test.py

数据集不上传，数据集命名要求：类别.序号.jpg,如cat.1.jpg,dog.1.jpg.....



