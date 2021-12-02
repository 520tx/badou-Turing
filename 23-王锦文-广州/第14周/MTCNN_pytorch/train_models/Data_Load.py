import sys
import torch
import torch.utils.data as data
import cv2
import os
from torchvision import transforms
from PIL import Image,ImageFile

def readDataLoader(label_file,batch_size,num_workers=1):
    print("label_file is {}".format(label_file))  
    f = open(label_file, 'r')
    samples=f.readlines()
    print("Total size of the dataset is: ", len(samples))
    data_transforms = {
        'train': transforms.Compose([
            transforms.ColorJitter(brightness=1,contrast=1,hue=0.5),# 随机从 0 ~ 2 之间亮度变化，1 表示原图
            transforms.RandomGrayscale(p=0.5),    # 以0.5的概率进行灰度化
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #transforms.Resize((img_size,img_size),3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }

    image_datasets=PNetDataSet(samples,data_transforms['train'])
    DataLoaderS = torch.utils.data.DataLoader(image_datasets,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=True) 
    return DataLoaderS,len(samples)

class PNetDataSet(data.Dataset):   
    def __init__(self, imgs, transforms=None):
        '''
        目标：获取所有图片路径，并根据训练、验证、测试划分数据
        '''
        self.imgs = imgs
        self.transforms = transforms

    #img torch.Size([batch_size, 3, 12, 12])--归一化的直接去训练
    #label tensor([ 0, -2,  0,  0,  1]) --batch_size
    #gt_bbox torch.Size([batch_size, 4])
    #gt_landmark: torch.Size([batch_size, 10])
    def __getitem__(self, index):
            '''
            返回一张图片的数据 (输入图片矩阵,该图片中gt的标签，gt的box坐标，landmark的坐标)
            '''
            line = self.imgs[index] 
            line=line.strip().split(" ")
            img_path=line[0]#../prepare_data/DATA/12/positive/343776.jpg
            # img=cv2.imread(img_path)#hwc bgr
            # img=img[...,[2,1,0]]#hwc rgb
            # print("img_path:",img_path)
            data = Image.open(img_path)
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            if len(data.getbands())!=3:
                print("{} is not RGB".format(img_path))
                data = data.convert('RGB')
            data = self.transforms(data)
            # data=data.unsqueeze(0)
            label = int(line[1])
            gt_bbox,gt_landmark=self.__gettensor__(line)
            # print(data.shape,gt_bbox.shape,gt_landmark.shape)
            return img_path,data, label,gt_bbox,gt_landmark
    def __gettensor__(self,info):
            gt_bbox_tensor=torch.zeros(4,dtype=torch.float32)
            gt_landmark_tensor=torch.zeros(10,dtype=torch.float32)
            if len(info) == 6:
                gt_bbox_tensor[0]=float(info[2])
                gt_bbox_tensor[1]=float(info[3])
                gt_bbox_tensor[2]=float(info[4])
                gt_bbox_tensor[3]=float(info[5])
            if len(info) == 12:
                # gt_bbox_tensor[0]=float(info[2])
                # gt_bbox_tensor[1]=float(info[3])
                # gt_bbox_tensor[2]=float(info[4])
                # gt_bbox_tensor[3]=float(info[5])
                gt_landmark_tensor[0]= float(info[2])
                gt_landmark_tensor[1]= float(info[3])
                gt_landmark_tensor[2] = float(info[4])
                gt_landmark_tensor[3] = float(info[5])
                gt_landmark_tensor[4] = float(info[6])
                gt_landmark_tensor[5] = float(info[7])
                gt_landmark_tensor[6]= float(info[8])
                gt_landmark_tensor[7]= float(info[9])
                gt_landmark_tensor[8] = float(info[10])
                gt_landmark_tensor[9] = float(info[11])  
            return gt_bbox_tensor,gt_landmark_tensor

    def __len__(self):
        '''
        返回数据集中所有图片的个数
        '''
        return len(self.imgs)

