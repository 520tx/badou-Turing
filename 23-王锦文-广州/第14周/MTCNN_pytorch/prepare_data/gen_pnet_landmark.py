import numpy as np
import numpy.random as npr
import os
import argparse
import cv2
from utils import IoU
from BBox_utils import BBox
from Landmark_utils import rotate, flip
import random
import sys
# 从lfw 数据集标注文件中trainImageList.txt中生成pnet/Rnet/onet网络的landmark数据
# 三个网络的landmark数据都是从trainImageList.txt文件中获得 不同的时生成txt文件的路径以及对应图片的路径以及做了shuffle之后数据的不同


# 负样本：0  并交比小于0.3
# 正样本：1  并交比大于0.65
# 部分脸：-1 并交比在0.4~0.65之间
parser = argparse.ArgumentParser(description="prepare landmark data for mtcnn pnet")
parser.add_argument('--lfw_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/lfw/train",
                    help="lfw landmark数据集的路径")
parser.add_argument('--lfw_file', type=str, default="trainImageList.txt",help="lfw landmark数据集数据文件 dataser格式为：路径 x1 x2 y1 y2 landmark坐标 10个")
parser.add_argument('--net_name', type=str, default="ONet",help="生成landmark的网络名称")
#parser.add_argument('--out_file_name', type=str, default="./DATA/12/landmark_12_aug.txt",help="12")
#parser.add_argument('--dstdir', type=str, default="./DATA/12/train_PNet_landmark_aug",help="pnet neg图片样本保存的目录")
args = parser.parse_args()



def show_bbox(img,corner_arr,land_marks):
    print(img.shape)

    list_corner=[corner_arr.left,corner_arr.top,corner_arr.right,corner_arr.bottom]
    cv2.rectangle(img,(list_corner[0],list_corner[1]),(list_corner[2],list_corner[3]),(255,0,0))

    for i in range(land_marks.shape[0]):
        landmark=land_marks[i,:]
        landmark=list(map(int,landmark))
        cv2.circle(img,(landmark[0],landmark[1]),1,(0,0,255))
        
    # cv2.namedWindow("bbox",0)    
    cv2.imshow("bbox",img)
    cv2.waitKey(2000)

#从lfw 数据集中获取数据
def getDataFromTxt(with_landmark=True):
    """
    Generate data from txt file
    return [(img_path, bbox, landmark)]
            bbox: [left, right, top, bottom]
            landmark: [(x1, y1), (x2, y2), ...]
    """
    with open(args.lfw_file, 'r') as fd:
        lines = fd.readlines()
    result = []
    for line in lines:
        components = line.strip().split(' ')
        img_path = os.path.join(args.lfw_dir, components[0]).replace('\\','/') # file path
        # bounding box, (x1, y1, x2, y2)
        #bbox = (components[1], components[2], components[3], components[4])
        bbox = (components[1], components[3], components[2], components[4])        
        bbox = [float(_) for _ in bbox]
        bbox = list(map(int,bbox))
        # landmark
        if not with_landmark:
            result.append((img_path, BBox(bbox)))
            continue
        landmark = np.zeros((5, 2))
        for index in range(0, 5):
            rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = rv
        #normalize
        result.append((img_path, BBox(bbox), landmark))
    return result

base_dir="/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA"
def GenerateData(argument=True):
    if args.net_name=="PNet":
        out_file_name=os.path.join(base_dir,"12/landmark_12_aug.txt")
        dstdir=os.path.join(base_dir,"12/landmark_aug")
        size=12
    elif args.net_name=="RNet":
        out_file_name=os.path.join(base_dir,"24/landmark_24_aug.txt")
        dstdir=os.path.join(base_dir,"24/landmark_aug")
        size=24
    elif args.net_name=="ONet":
        out_file_name=os.path.join(base_dir,"48/landmark_48_aug.txt")
        dstdir=os.path.join(base_dir,"48/landmark_aug")
        size=48
    else:
        print("the net name error,",args.net_name)
        return
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    # get image path , bounding box, and landmarks from file 'ftxt'
    data = getDataFromTxt()
    f = open(out_file_name,'w')
    image_id = 0
    idx = 0
    print("the num of img is ",len(data))
    for (imgPath, bbox, landmarkGt) in data:
        F_imgs = []#每张图片对应的face正样本 矩阵数据--包括gt iou>0.65 的box 增强后的box
        F_landmarks = []#每张图片对应的兰黛land_mark 包括gt's landmark 以及 iou>0.65 landmark 增强后的land_mark
        img=cv2.imread(imgPath)
        #*******************关键点验证××××××××××××××××××××××
        # show_bbox(img,bbox,landmarkGt)
        # continue
        #*******************关键点验证××××××××××××××××××××××
        assert (img is not None),"图片名称为{}".format(imgPath)
        img_h,img_w,img_c = img.shape
        gt_box = np.array([bbox.left,bbox.top,bbox.right,bbox.bottom])#------->>>>gt 的框（x1,y1,x2,y2）
        #get sub-image from bbox  left=x1 top=y1 right=x2 bottom=y2
        f_face = img[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
        # resize the gt image to specified size
        f_face = cv2.resize(f_face,(size,size))#把人脸gt resize (12,12)
        #initialize the landmark
        landmark = np.zeros((5, 2))
        #normalize land mark by dividing the width and height of the ground truth bounding box
        # landmakrGt is a list of tuples
        for index,markpoint in enumerate(landmarkGt):
            # (( x - bbox.left)/ width of bounding box, (y - bbox.top)/ height of bounding box
            x_r=(markpoint[0]-gt_box[0])/(gt_box[2]-gt_box[0])
            y_r=(markpoint[1]-gt_box[1])/(gt_box[3]-gt_box[1])
            landmark[index]=(x_r,y_r)
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(10))
        if argument:#需要数据增强的话
            idx = idx + 1
            if idx % 100 == 0:
                print(idx, "images done")
            x1, y1, x2, y2 = gt_box
            #gt's width
            gt_w = bbox.w
            #gt's height
            gt_h = bbox.h 
            #去掉比较小的人脸区域      
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            
            for i in range(10):
                #将gt box进行偏移同时w,h进行变换后得到的box
                bbox_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                nx1 = int(max(x1+gt_w/2-bbox_size/2+delta_x,0))
                ny1 = int(max(y1+gt_h/2-bbox_size/2+delta_y,0))
                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size
                if nx2 > img_w or ny2 > img_h:
                    continue
                crop_box = np.array([nx1,ny1,nx2,ny2])
                #cal iou
                iou = IoU(crop_box, np.expand_dims(gt_box,0))
                cropped_im = img[ny1:ny2+1,nx1:nx2+1,:]#作变化后 bbox对应的图片
                resized_im = cv2.resize(cropped_im, (size, size))
                if iou > 0.65:#iou>0.65作为正样本
                    #random shift  一张脸随机进行编译以及进行图像旋转反转等变化
                    F_imgs.append(resized_im)
                    #normalize   平移后的框左上角(nx1，ny1)  gt框左上角（one[0]，one[1]） 变化后的框长宽 bbox_size
                    for index, markpoint in enumerate(landmarkGt):
                        rv = ((markpoint[0]-nx1)/bbox_size, (markpoint[1]-ny1)/bbox_size)
                        landmark[index] = rv
                    F_landmarks.append(landmark.reshape(10))
                    landmark = np.zeros((5, 2))
                    landmark_ = F_landmarks[-1].reshape(-1,2)#取最后一个元素，并reshape
                    bbox = BBox([nx1,ny1,nx2,ny2])  
                    #mirror                    
                    if random.choice([0,1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        #c*h*w
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    #rotate
                    if random.choice([0,1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), 5)#逆时针旋转
                        #landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))
                
                        #flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))                
                    
                    #anti-clockwise rotation
                    if random.choice([0,1]) > 0: 
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), -5)#顺时针旋转
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))
                
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))   

        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
        #print(F_imgs.shape)#(1, 12, 12, 3)
        #print(F_landmarks.shape)#(1, 10)
        #sys.exit()
        count=0
        for i in range(len(F_imgs)):
            if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                #print("$$$$$$$$  {}".format(np.where(F_landmarks[i] <= 0, 1, 0)))
                continue
            if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                # print("&&&&&&&&&  {}".format(F_landmarks[i]))
                # print("&&&&&&&&&  {}".format(np.where(F_landmarks[i] >= 1, 1, 0)))
                continue
            count+=1
            cv2.imwrite(os.path.join(dstdir,"%d.jpg" %(image_id)), F_imgs[i])
            landmarks = map(str,list(F_landmarks[i]))
            f.write(os.path.join(dstdir,"%d.jpg" %(image_id))+" -2 "+" ".join(landmarks)+"\n")
            image_id = image_id + 1
            #landmark_12_aug.txt 的文件格式为  图片名称 -2 x1 y1 x2 y2 x3 y3 x4 y4 x5 y5(做了归一化的)
            # (( x - bbox.left)/ width of bounding box, (y - bbox.top)/ height of bounding box
    print("PNet landmark data size:",image_id)#347283
    f.close()
    # return F_imgs,F_landmarks        

def main():
    GenerateData()
    

if  __name__ == "__main__":
    main()