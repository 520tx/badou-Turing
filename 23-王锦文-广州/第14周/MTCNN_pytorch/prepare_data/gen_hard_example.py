import argparse
import pickle as pickle
import os
import sys
sys.path.insert(0,"..")
from train_models.Mtcnn_Net import PNet,RNet 
import torch
import torch.nn as nn
from BBox_utils  import *
from prepare_data.utils import *
from Detection.Mtcnn_Detector import MtcnnDetector
from multiprocessing import Pool
# from models import PNet
# from Nets import PNet




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='生成R网络和O网络的数据',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--wider_path', help='',default='/media/ubuntu_data2/02_dataset/wjw/dataset/WIDER', type=str)
parser.add_argument('--wider_file_path', help='',default='./', type=str)
parser.add_argument('--net_name', help='test net type, can be pnet, rnet or onet',default='PNet', type=str)
parser.add_argument('--base_dir', help='',default='/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/', type=str)
parser.add_argument('--save_pickle_path', help='test net type, can be pnet, rnet or onet',default='/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/48_1/RNet', type=str)
parser.add_argument('--load_pnet_path', help='test net type, can be pnet, rnet or onet',default='../train_models/Pnet_weight', type=str)
parser.add_argument('--load_rnet_path', help='test net type, can be pnet, rnet or onet',default='../train_models/Rnet_weight', type=str)
parser.add_argument('--load_onet_path', help='test net type, can be pnet, rnet or onet',default='../train_models/Onet_weight', type=str)

#parser.add_argument('--prefix', dest='prefix', help='prefix of model name', default=['../train_models/Pnet_weight', '../train_models/Rnet_weight', '../train_models/Onet_weight'],type=str)
#parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load',default=[19, 19, 19], type=int)
#parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction',default=[2048, 256, 16], type=int)
parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet',default=[0.3, 0.15, 0.7], type=float)
parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',default=20, type=int)
parser.add_argument('--stride', dest='stride', help='stride of sliding window',default=2, type=int)
parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
parser.add_argument('--shuffle', dest='shuffle', help='shuffle data on visualization', action='store_true')
parser.add_argument('--use_gpu', type=int, default=1,help='use gpu')
parser.add_argument('--gpus', type=str, default='0')
args = parser.parse_args()

def load_model(net_name,model_path):
    ret=False
    model=None
    if net_name=="PNet":
        model=PNet()
    elif net_name=="RNet":
        model=RNet()
    model_dict =  model.state_dict()

    # if net_name=="RNet":
    #     for param in model_dict.keys():
    #         print(model_dict[param])

    if os.path.isfile(model_path):
        print("the model file path:",model_path)
        state_dict=torch.load(model_path)
        print(state_dict['epoch'])
        # print(state_dict['epoch_acc'])
        # params=model.state_dict() 
        params=state_dict["model_state_dict"] 
        # state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
        state_dict1 = {k: v for k, v in params.items() if k in model_dict.keys()}#richard 1126 params不是state_dict
        for (k1,v1),(k2,v2) in zip(params.items(),model_dict.items()):
            print(k1,k2)
       
        # sys.exit()
        model_dict.update(state_dict1)
        model.load_state_dict(model_dict)
        print("load cls model successfully")
        ret=True
        # sys.exit()
    else:
        print("the path:{} is not a file".format(args.load_path))
        sys.exit()

    # 是否使用gpu
    if args.use_gpu:
        model = model.cuda()
        gpu_id=args.gpus.strip().split(',')
        print("gpu_id,",len(gpu_id))
        if len(gpu_id) > 1:
            model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])
    model.eval()
    return ret,model
    
#net : 24(RNet)/48(ONet)
#data: dict()
def save_hard_example(data,save_path):
    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image
    data_base=args.base_dir
    if args.net_name=="PNet":#P网络生成R网络的数据，R网络的图片大小为24
        img_size=24
        neg_dir = data_base+'/24/negative'
        pos_dir = data_base+'/24/positive'
        part_dir= data_base+'/24/part'
        neg_label_file = data_base+"/24/neg_24.txt"
        pos_label_file = data_base+"/24/pos_24.txt"
        part_label_file = data_base+"/24/part_24.txt" 

    if args.net_name=="RNet":
        img_size=48
        neg_dir = data_base+'/48_1/negative'
        pos_dir = data_base+'/48_1/positive'
        part_dir= data_base+'/48_1/part'
        neg_label_file = data_base+"/48_1/neg_48.txt"
        pos_label_file = data_base+"/48_1/pos_48.txt"
        part_label_file = data_base+"/48_1/part_48.txt" 

   
    #create dictionary shuffle   
    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)

    print("processing %d images in total" % num_of_images)

    # save files
    
    neg_file = open(neg_label_file, 'w')
    pos_file = open(pos_label_file, 'w')
    part_file = open(part_label_file, 'w')
    #read detect result
    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb'))
    # print(len(det_boxes), num_of_images)
    print(len(det_boxes))
    print(num_of_images)
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    per_n_idx = 0
    per_p_idx = 0
    per_d_idx = 0
    image_done = 0
    #im_idx_list image index(list)       im_idx：string:img_path
    #det_boxes pnet detect result(list)  dets：[num,5] num:图片img_path预测为人脸的boundingbox  x1,y1,x2,y2,score
    #gt_boxes_list gt(list)              gts：[N,4]   num:图片img_pathGT  x1,y1,x2,y2
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if image_done+1 % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1
        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        #change to square
        dets = convert_to_square(dets)#[num 5]
        dets[:, 0:4] = np.round(dets[:, 0:4])# 取出boundingbox 的坐标[num 4]
        neg_num = 0

        ###########################
        # 验证预测结果与gt的关系
        # im_show=img.copy()
        # cv2.imshow("mtcnn",im_show)
        # print(dets.shape)
        # # print(boxes[0,1])
        # cv2.waitKey(10)
        # for i in range(gts.shape[0]):
        #     x1=int(max(gts[i,0],0))
        #     y1=int(max(gts[i,1],0))
        #     x2=int(min(gts[i,2],im_show.shape[1]))
        #     y2=int(min(gts[i,3],im_show.shape[0]))
        #     cv2.rectangle(im_show,(x1,y1),(x2,y2),(0,255,0),1)
        # for i in range(dets.shape[0]):
        #     x_left, y_top, x_right, y_bottom, _ = dets[i].astype(int)
        #     width = x_right - x_left + 1
        #     height = y_bottom - y_top + 1
        #     if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
        #         continue
        #     Iou = IoU(dets[i], gts)
        #     x1=int(max(dets[i,0],0))
        #     y1=int(max(dets[i,1],0))
        #     x2=int(min(dets[i,2],im_show.shape[1]))
        #     y2=int(min(dets[i,3],im_show.shape[0]))
        #     if dets[i,4]>=0.5:
        #         cv2.rectangle(im_show,(x1,y1),(x2,y2),(0,0,255),1)
        #     else:
        #         cv2.rectangle(im_show,(x1,y1),(x2,y2),(255,0,0),1)
        #     cv2.imshow("mtcnn",im_show)
        #     if cv2.waitKey(1) == ord('q'):
        #             break
        # print("draw finish")
        # cv2.waitKey(8000)
        ###########################



        #*********
        # 一张图片可得到 num较多，假设bounding box num=500,GT N=5 500个中可能有480个bounding box与GT的IOU值<0.3 10个>0.65 10个>0.4
        # 所以一张图片得到 60 neg 较少的pos 与较少的part
        #一个图片有num个 bounding box 计算每个bounding box与该图片N个GT的IOU,
        # IOU最大值 <0.3 从图片中扣出bounding box区域保存作为neg
        # IOU最大值 >=0.6 从图片中扣出bounding box区域保存作为pos，同时计算bounding box相对于GT的offset，用于保存
        #I OU最大值 >=0.45 从图片中扣出bounding box区域保存作为pos，同时计算bounding box相对于GT的offset，用于保存
        #********
        for box in dets:#循环一张图片所有的预测的bounding box
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue
            
            #计算当前boundingbox 与本张图上所有GT的IOU
            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)#(N, ) 每个元素为该boundingbox 与N个GT的iou
            #print(np.)
            #从原图中扣出该boundingbox 所在的区域，并resize成R网络或者O网络所需要的图片大小
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (img_size, img_size),interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            # Iou with all gts must below 0.3            
            if np.max(Iou) < 0.3 and neg_num < 60:
                #save the examples
                save_file = os.path.join(neg_dir, "%s.jpg" % n_idx)
                # print(save_file)
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
                per_n_idx+=1
                ###########################
                # 绘制当前box与gt的关系
                # x1=int(max(box[0],0))
                # y1=int(max(box[1],0))
                # x2=int(min(box[2],im_show.shape[1]))
                # y2=int(min(box[3],im_show.shape[0]))
                # cv2.rectangle(im_show,(x1,y1),(x2,y2),(0,0,255),1)
                # cv2.imshow("mtcnn",im_show)
                # if cv2.waitKey(1) == ord('q'):
                #         break
                ###########################
            else:
                # find gt_box with the highest iou
                #找到该bounding box 与GT有的最大IOU值的索引
                idx = np.argmax(Iou)
                Idx=np.argsort(Iou)
                # print(Iou[Idx])
                #找到该GT，获取该GT的坐标
                x1, y1, x2, y2 = gts[idx]

                # compute bbox reg label
                #计算该boundingbox与GT的offset
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                #该bounding box 与GT有的最大IOU值 >0.65 pos >=0.45 part
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                    per_p_idx+=1
                    ###########################
                    # 绘制当前box与gt的关系
                    # x1=int(max(box[0],0))
                    # y1=int(max(box[1],0))
                    # x2=int(min(box[2],im_show.shape[1]))
                    # y2=int(min(box[3],im_show.shape[0]))
                    # cv2.rectangle(im_show,(x1,y1),(x2,y2),(255,0,0),1)
                    # cv2.imshow("mtcnn",im_show)
                    # if cv2.waitKey(1) == ord('q'):
                    #         break
                    ###########################

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
                    per_d_idx+=1
                    ###########################
                    # 绘制当前box与gt的关系
                    # x1=int(max(box[0],0))
                    # y1=int(max(box[1],0))
                    # x2=int(min(box[2],im_show.shape[1]))
                    # y2=int(min(box[3],im_show.shape[0]))
                    # cv2.rectangle(im_show,(x1,y1),(x2,y2),(255,255,0),1)
                    # cv2.imshow("mtcnn",im_show)
                    # if cv2.waitKey(1) == ord('q'):
                    #         break
                    ###########################
        print("{} gen {} pos {} neg {} part".format(im_idx,per_p_idx,per_n_idx,per_d_idx))
        per_n_idx = 0
        per_p_idx = 0
        per_d_idx = 0
    print("经过网络后得到pos:{} neg:{} part:{}".format(p_idx,n_idx,d_idx))
    neg_file.close()
    part_file.close()
    pos_file.close()

def test_net(thresh=[0.6, 0.6, 0.7], min_face_size=25):
    print("the thresh: ",thresh)
    print("the name of current model tested: ", test_net)
    #获得模型路径
    #model_path=["{}/checkpoints_epoch_{}.tar".format(i,j) for i,j in zip(args.prefix,args.epoch)]
    print("the pnet model_path ",args.load_pnet_path)

    print("**********start load pnet***********")
    ret,p_model=load_model("PNet",args.load_pnet_path)
    if not ret:
        return
    print("**********load pnet successfully***********")

    if args.net_name=="RNet":
        print("**********start load Rnet***********")
        ret,r_model=load_model("RNet",args.load_rnet_path)
        if not ret:
            return
        print("**********load Rnet successfully***********")
    else:
        r_model=None
    #read_annotation
    data=read_annotation(args.wider_path,"./wider_face_train_bbx_gt.txt")
    # data=readWider(args.wider_path,"./wider_face_train.txt")
    #dataloader,datasize=readImgData(data=data['images'],batch_size=1,shuffle=False,num_workers=3)

    print('load test data')
    test_data = TestLoader(data['images'])
    print ('finish loading')


    #创建检测器对象
    mtcnn_detector = MtcnnDetector(pnet=p_model,rnet=r_model, min_face_size=min_face_size, threshold=thresh,use_gpu=args.use_gpu)
    # detections,_ = mtcnn_detector.detect_face(dataloader,datasize)
    detections,_ = mtcnn_detector.detect_face(test_data,test_data.size)
    print ('finish detecting ')

    save_path=args.save_pickle_path
    print ('save_path is :',save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_file = os.path.join(save_path, "detections.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(detections, f,1)
    print("%s 网络测试完成开始OHEM" % args.net_name)
    save_hard_example(data, save_path)

#python  gen_hard_example.py  --net_name RNet --load_pnet_path ../train_models/PNet_debug/checkpoints_epoch_28.tar ../train_models/RNet_weight/iter_41249.pth.tar
def main():
    test_net(thresh=args.thresh, min_face_size=args.min_face)

if __name__=='__main__':
    main()