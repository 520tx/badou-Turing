import os
from os.path import join, exists
import time
import cv2
import numpy as np
from torchvision import transforms
import torch
import torch.utils.data as data
from PIL import Image,ImageFile


def read_annotation(base_dir, label_path):
    """
    read label file
    :param dir: path
    :return:
    """
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # image path
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/WIDER_train/images/' + imagepath
        assert os.path.isfile(imagepath),"{} is not a file".format(imagepath)
        images.append(imagepath)
        # face numbers
        nums = labelfile.readline().strip('\n')
        # im = cv2.imread(imagepath)
        # h, w, c = im.shape
        one_image_bboxes = []
        for i in range(int(nums)):
            # text = ''
            # text = text + imagepath
            bb_info = labelfile.readline().strip('\n').split(' ')
            # only need x, y, w, h
            face_box = [float(bb_info[i]) for i in range(4)]
            # text = text + ' ' + str(face_box[0] / w) + ' ' + str(face_box[1] / h)
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]
            # text = text + ' ' + str(xmax / w) + ' ' + str(ymax / h)
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
            # f.write(text + '\n')
        bboxes.append(one_image_bboxes)


    data['images'] = images#all images
    data['bboxes'] = bboxes#all image bboxes
    # f.close()
    return data

#获取Wider数据集数据 
#0--Parade/0_Parade_marchingband_1_849 448.51 329.63 570.09 478.23 
#0--Parade/0_Parade_Parade_0_904 360.92 97.92 623.91 436.46 
#base_dir wider-face数据集的基础路径 路径下有WIDER_train/WIDER_val/WIDER_test/文件夹
#return dic key:图片路径 value:numpy数组 shape=(N,4)N为该图片中包含人脸的个数
def readWider(base_dir,file_path):
    """
    read label file
    :param dir: path
    :return:
    """
    data = dict()
    images = []
    bboxes = []
    labelfile = open(file_path, 'r')
    lines=labelfile.readlines()
    for line in lines[0:50]:
        # image path
        line =line.strip().split(' ')
        imagepath=line[0]+".jpg"
        imagepath = base_dir + '/WIDER_train/images/' + imagepath
        assert os.path.isfile(imagepath),"{} is not a file".format(imagepath)
        images.append(imagepath)
        bbox = list(map(float, line[1:]))
        boxes=np.array(bbox,dtype=np.float32).reshape(-1,4)#gt
        bboxes.append(boxes)
    data['images'] = images#all images path
    data['bboxes'] = bboxes#all image bboxes
    # f.close()
    return data

#txt:lfw 数据集的文件
def getDataFromTxt(txt,data_path, with_landmark=True):
    pass

#从data中读取batch_size个数据 data为[]类型
def readImgData(data,batch_size,shuffle=False,num_workers=1):
    print("Total size of the dataset is: ", len(data))
    # if net_name=="pnet":
    #     img_size=12
    #     current_scale = float(img_size) / min_face  # find initial scale
    #     data_transforms =transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_datasets=ReadImgDataFromList(data)
    DataLoaderS = torch.utils.data.DataLoader(image_datasets,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers) 
    return DataLoaderS,len(data)

class BBox(object):
    """
        Bounding Box of face
    """
    def __init__(self, bbox):
        self.left = bbox[0]
        self.top = bbox[1]
        self.right = bbox[2]
        self.bottom = bbox[3]
        
        self.x = bbox[0]
        self.y = bbox[1]
        self.w = bbox[2] - bbox[0]+1
        self.h = bbox[3] - bbox[1]+1

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)
    #offset
    def project(self, point):
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x, y])
    #absolute position(image (left,top))
    def reproject(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])
    #landmark: 5*2
    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p
    #change to offset according to bbox
    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p
    #f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
    #self.w bounding-box width
    #self.h bounding-box height
    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])

class TestLoader:
    #imdb image_path(list)
    def __init__(self, imdb, batch_size=1, shuffle=False):
        self.imdb = imdb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(imdb)#num of data
        #self.index = np.arange(self.size)
        
        self.cur = 0
        self.data = None
        self.label = None

        self.reset()
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            #shuffle test image
            np.random.shuffle(self.imdb)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size
    #realize __iter__() and next()--->iterator
    #return iter object
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        imdb = self.imdb[self.cur]
        '''
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        #picked image
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        # print(imdb)
        '''
        #print type(imdb)
        #print len(imdb)
        #assert len(imdb) == 1, "Single batch only"
        im = cv2.imread(imdb)
        self.data = im

class ReadImgDataFromList(data.Dataset):   
    def __init__(self, ImgPathList,transforms=None):
        '''目标：获取路径下的图片数据'''
        self.imgs = ImgPathList
        self.transforms = transforms

    #return torch.tensor (batchsize,3,h,w)  ---归一化了 可直接用于预测
    #******eturn torch.tensor (1,h,w,3)  ---未作任何处理
    def __getitem__(self, index):
        img_path = self.imgs[index] #./DATA/12/positive/343776.jpg 
        # img_path="../prepare_data/"+line[0][1:]#../prepare_data/DATA/12/positive/343776.jpg
        print("img_path:",img_path)
        data=cv2.imread(img_path,1)
        print(data.shape)
        print(data[0,0,:])
        # height, width, channels = data.shape
        # new_height = int(height * self.scale)  # resized new height
        # new_width = int(width * self.scale)  # resized new width
        # new_dim = (new_width, new_height)
        # img_resized = cv2.resize(data, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image hwc bgr
        # img_resized=img_resized[...,[2,1,0]]#h w c bgr to hwc rgb
        # data = self.transforms(img_resized)
        return img_path,data  #hwc gbr

    #return PIL数据类型(没有做任何变换)
    # def __getitem__(self, index):
    #         '''
    #         返回一张图片的数据 (输入图片矩阵)
    #         '''
    #         img_path = self.imgs[index] #./DATA/12/positive/343776.jpg 
    #         # img_path="../prepare_data/"+line[0][1:]#../prepare_data/DATA/12/positive/343776.jpg
    #         print("img_path:",img_path)
    #         data = Image.open(img_path)
    #         ImageFile.LOAD_TRUNCATED_IMAGES = True
    #         if len(data.getbands())!=3:
    #         #   print("{} is not RGB".format(img_path))
    #             data = data.convert('RGB')
    #         #TODO 先转成tensor不做归一化等操作
    #         data = self.transforms(data)
    #         return data
    def __len__(self):
        '''
        返回数据集中所有图片的个数
        '''
        return len(self.imgs)


# def main():
#     # data_transforms =transforms.Compose([
#     #         transforms.ToTensor()])
#     data=readWider('/media/ubuntu_data2/02_dataset/wjw/dataset/WIDER',"./wider_face_train.txt")
#     print(data['images'][0])
#     dataload,datasize=readImgData(data=data['images'],batch_size=1)
#     for i,img in enumerate(dataload):
#         print(i)
#         if i==3:
#             break
#         print("img",img.shape)

# if __name__ == "__main__":
#     main()