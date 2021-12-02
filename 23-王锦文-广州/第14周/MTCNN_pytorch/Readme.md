一.所用文件说明：
1.wider_face_train.txt文件
1）格式:路径 x1 y1 x2 y2 
2）例：
0--Parade/0_Parade_marchingband_1_849 448.51 329.63 570.09 478.23 
0--Parade/0_Parade_Parade_0_904 360.92 97.92 623.91 436.46 
3）使用地方：
在gen_pnet_data.py用于生成p网络pos/neg/part的样本


2.trainImageList.txt文件---lfw landmark数据集数据标注文件
1）格式:路径 x1 x2 y1 y2 landmark坐标 10个
2）例：lfw_5590\Aaron_Eckhart_0001.jpg 84 161 92 169 106.250000 107.750000 146.750000 112.250000 125.250000 142.750000 105.250000 157.750000 139.750000 161.750000
3）使用：在gen_pnet_landmark.py用于生成p网络landmark的样本

3.wider_face_train_bbx_gt.txt文件
1）.格式:
路径 
图片人脸box数量
x1 y1 x2 y2 
x1 y1 x2 y2 

例：
0--Parade/0_Parade_marchingband_1_849.jpg(wider_face_train文件中没有.jpg)
1
449 330 122 149 0 0 0 0 0 0 

2）与wider_face_train.txt文件类似，可以不用该文件



二.准备PNet的数据
1.python gen_pnet_data.py 生成
parser.add_argument('--pnet_posdata_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/12/pos_12.txt",help="pnet pos 图片路径文件所在的路径")
parser.add_argument('--pnet_negdata_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/12/neg_12.txt",help="pnet neg 图片路径文件所在的路径")
parser.add_argument('--pnet_partdata_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/12/part_12.txt",help="pnet part 图片路径文件所在的路径")
parser.add_argument('--pnet_posimg_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/12/positive",help="pnet pos图片样本保存的目录")
parser.add_argument('--pnet_negimg_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/12/negative",help="pnet neg图片样本保存的目录")
parser.add_argument('--pnet_partimg_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/12/part",help="pnet part图片样本保存的目录")

2.python gen_pnet_landmark.py 生成landmark文件
生成的文件和图片路径分别保存在
/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/12/landmark_12_aug.txt
/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/12/landmark_aug

3.python gen_pnet_imglist.py 生成
将pos_12.txt,neg_12.txt,part_12.txt,landmark_12_aug.txt文件中按照1:3:1:1抽取一部分保存在imglist_12.txt文件中

三.训练P网络
运行：
python train.py --batch_size 2560 --save_path ./PNet_weight/ --lr 0.001 --print_freq 100

模型路径：./train_models/PNet_weight/

四.生成rnet训练所需要的数据
1.python gen_hard_example.py --load_pnet_path ../train_models/PNet_weight/checkpoints_epoch_29.tar
pos:119315 neg:772738 part:588640
生成了pos_24.txt,neg_24.txt,part_24.txt
路径：/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/24

2.python gen_pnet_landmark.py  --net_name RNet
生成的文件和图片路径分别保存在
/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/24/landmark_24_aug.txt
/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/24/landmark_aug
PNet landmark data size: 178522

五.训练rnet
python train.py --net_name=RNet  --lr  0.001 --save_path ./Rnet_weight/ --num_epoch 12 --print_freq 100

python train.py --net_name ONet --save_path ./Onet_weight/ --lr 0.001 --num_epoch 12 --print_freq 100


生成R网络训练所需要的landmark数据--与P网络一样，都是从lfw数据集中标注文件中加载
运行脚本：python  gen_pnet_landmark.py --net_name RNets --load_pnet_path 



六.通过pnet rnet生成onet训练所需要的样本
python  gen_hard_example.py  --net_name RNet --load_pnet_path ../train_models/PNet_debug/checkpoints_epoch_29.tar --load_rnet_path ../train_models/Rnet_weight/checkpoints_iter_5615.tar
七.测试
在train_models目录下的test.py.执行python test.py得到结果
python  test.py  --data_dir ./img --load_pnet_path ../train_models/PNet_weight/checkpoints_epoch_29.tar --load_rnet_path ../train_models/Rnet_weight/checkpoints_iter_5615.tar --load_onet_path ../train_models/Onet_weight/checkpoints_iter_5615.tar
参考code：
https://github.com/AITTSMD/MTCNN-Tensorflow
由于训练时间关系，只训练了少量的epoch，因此效果不一定很准确。



