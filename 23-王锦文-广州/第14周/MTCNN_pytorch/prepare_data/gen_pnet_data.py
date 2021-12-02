import numpy as np
import numpy.random as npr
import os
import argparse
import cv2
import utils

# 负样本：0  并交比小于0.3
# 正样本：1  并交比大于0.65
# 部分脸：-1 并交比在0.4~0.65之间
parser = argparse.ArgumentParser(description="prepare images data for mtcnn pnet")
parser.add_argument('--pnet_posdata_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/torch_MTCNN_DATA/12/pos_12.txt",help="pnet pos 图片路径文件所在的路径")
parser.add_argument('--pnet_negdata_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/torch_MTCNN_DATA/12/neg_12.txt",help="pnet neg 图片路径文件所在的路径")
parser.add_argument('--pnet_partdata_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/torch_MTCNN_DATA/12/part_12.txt",help="pnet part 图片路径文件所在的路径")
parser.add_argument('--pnet_posimg_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/torch_MTCNN_DATA/12/positive",help="pnet pos图片样本保存的目录")
parser.add_argument('--pnet_negimg_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/torch_MTCNN_DATA/12/negative",help="pnet neg图片样本保存的目录")
parser.add_argument('--pnet_partimg_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/torch_MTCNN_DATA/12/part",help="pnet part图片样本保存的目录")
parser.add_argument('--anno_file', type=str, default="wider_face_train.txt",help="wider_face dataser格式为：路径 x1 y1 x2 y2")
parser.add_argument('--wider_dir', type=str, default="/media/ubuntu_data2/02_dataset/wjw/dataset/WIDER/WIDER_train/images",help="wider img path")

args = parser.parse_args()


def main():
    if not os.path.exists(args.pnet_posimg_dir):
        os.makedirs(args.pnet_posimg_dir)
    if not os.path.exists(args.pnet_negimg_dir):
        os.makedirs(args.pnet_negimg_dir)
    if not os.path.exists(args.pnet_partimg_dir):
        os.makedirs(args.pnet_partimg_dir)
    f_pos=open(args.pnet_posdata_dir,'w')
    f_neg=open(args.pnet_negdata_dir,'w')
    f_part=open(args.pnet_partdata_dir,'w')
    with open(args.anno_file, 'r') as f:#
        annotations = f.readlines()
    num = len(annotations)
    print("%d pics in total" % num)
    idx=0#处理的图片的个数
    n_idx=0#初始化负样本的个数
    p_idx=0
    d_idx=0
    for annotation in annotations:#0--Parade/0_Parade_marchingband_1_849 448.51 329.63 570.09 478.23
        annotation=annotation.strip().split(' ')
        #image path
        im_path = annotation[0]
        bbox = list(map(float, annotation[1:]))
        boxes=np.array(bbox,dtype=np.float32).reshape(-1,4)#gt
        # print(os.path.join(args.wider_dir,im_path+".jpg"))
        img=cv2.imread(os.path.join(args.wider_dir,im_path+".jpg"))#load image
        height, width, channel = img.shape
        neg_num_per = 0
        #处理图片个数+1
        idx+=1
        # keep crop random parts, until have 50 negative examples
        # get 50 negative sample from every image
        while neg_num_per<60:
            # size is a random number between 12 and min(width,height)/2
            size = npr.randint(12, min(width, height) / 2)
             #top_left coordinate
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
             #random crop
            crop_box = np.array([nx,ny,nx+size,ny+size])
             #calculate iou
            Iou = utils.IoU(crop_box, boxes)
            if np.max(Iou)<0.3:#负例 <0.3 resize为12*12 保存该crop图片为负例
                #crop a part from inital image
                cropped_im = img[ny : ny + size, nx : nx + size, :]
                #resize the cropped image to size 12*12
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                save_name = os.path.join(args.pnet_negimg_dir, "%s.jpg"%n_idx)
                f_neg.write(save_name + ' 0\n')
                cv2.imwrite(save_name, resized_im)
                n_idx += 1
                neg_num_per += 1
        #for every bounding boxes after getting 50 negative for each img
        for box in boxes:
            x1, y1, x2, y2 = box
            #gt's width
            w = x2 - x1 + 1
            #gt's height
            h = y2 - y1 + 1
            # ignore small faces and those faces has left-top corner out of the image
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 20 or x1 < 0 or y1 < 0:
                continue
            # crop another 5 images near the bounding box if IoU less than 0.3, save as negative samples
            for i in range(40):
                #size of the image to be cropped
                size = npr.randint(12, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                # max can make sure if the delta is a negative number , x1+delta_x >0
                # parameter high of randint make sure there will be intersection between bbox and cropped_box
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                # max here not really necessary
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                # if the right bottom point is out of image then skip
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = utils.IoU(crop_box, boxes)
                if np.max(Iou) < 0.3:
                     #crop a part from inital image
                    cropped_im = img[ny1 : ny1 + size, nx1 : nx1 + size, :]
                    #resize the cropped image to size 12*12
                    resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                    save_name = os.path.join(args.pnet_negimg_dir, "%s.jpg"%n_idx)
                    f_neg.write(save_name + ' 0\n')
                    cv2.imwrite(save_name, resized_im)
                    n_idx += 1
            # print("{} 生成{}个负样本结束".format(im_path,n_idx))
        #generate positive examples and part faces
            for i in range(60):
                # pos and part face size [minsize*0.8,maxsize*1.25]
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                # delta here is the offset of box center
                if w<5:
                    print (w)
                    continue
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)
                #show this way: nx1 = max(x1+w/2-size/2+delta_x)
                # x1+ w/2 is the central point, then add offset , then deduct size/2
                # deduct size/2 to make sure that the right bottom corner will be out of
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                #show this way: ny1 = max(y1+h/2-size/2+delta_y)
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size
                if nx2 > width or ny2 > height:
                    continue 
                #calculate iou
                crop_box = np.array([nx1, ny1, nx2, ny2])
                box_ = box.reshape(1, -1)
                iou = utils.IoU(crop_box, box_)
                #yu gt de offset
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
                #crop
                cropped_im = img[ny1 : ny2, nx1 : nx2, :]
                #resize
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                if iou  >= 0.65:
                    save_name = os.path.join(args.pnet_posimg_dir, "%s.jpg"%p_idx)
                    #最终pos_12.txt的文件格式为： 文件名 分类标签(part:-1 neg:0 pos:1) x1偏移((x1 - nx1) / float(size))  y1偏移   x2偏移   y2偏移 
                    #当分类标签为0时，后面没有偏移
                    f_pos.write(save_name + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_name, resized_im)
                    p_idx += 1
                elif iou >= 0.4:
                    save_name = os.path.join(args.pnet_partimg_dir, "%s.jpg"%d_idx)
                    f_part.write(save_name + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_name, resized_im)
                    d_idx += 1
            # box_idx += 1
        if idx % 100 == 0:
            print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))
    f_pos.close()
    f_neg.close()
    f_part.close()
    #12800 images done, pos: 457862 part: 1125680 neg: 995395
    #12800 images done, pos: 1371377 part: 3379816 neg: 3612021

if  __name__ == "__main__":
    main()