#coding:utf-8
from Mtcnn_Net import *
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import sys
from Data_Load import *
import datetime
from torch.optim import lr_scheduler
import shutil
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       #


parser = argparse.ArgumentParser(description="PyTorch implementation of mtcnn")
#parser.add_argument('--train_data_dir', type=str, default='../prepare_data/DATA/imglists/PNet')
parser.add_argument('--train_data_dir', type=str, default='/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/12/')
parser.add_argument('--net_name', type=str, default='PNet')
parser.add_argument('--data_dir', type=str, default='/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/')
parser.add_argument('--batch_size', type=int, default=384)
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--net_width', type=float, default=1.0)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--save_epoch_freq', type=int, default=2)
parser.add_argument('--save_path', type=str, default="./Pnet_weight/")
parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
parser.add_argument('--start_epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
parser.add_argument('--load_path', type=str, default="", help="For training from one model_file")
parser.add_argument('--momentum', default=0.9, type=float,help='Momentum value for optim')
parser.add_argument('--weight_decay', default=0.00004, type=float,help='Weight decay for SGD')
args = parser.parse_args()



def get_loss_ratio(net_name):
    if net_name == 'PNet':#对于P-Net 和R-Net 则更关注检测框定位的准确性，而较少关注关键点定位的准确性，所以关键点定位损失权重较小
        radio_cls_loss = 1
        radio_bbox_loss = 0.5
        radio_landmark_loss = 0.5
    elif net_name == 'RNet':
        radio_cls_loss = 1.0
        radio_bbox_loss = 0.5
        radio_landmark_loss = 0.5
    else:
        radio_cls_loss = 1.0
        radio_bbox_loss = 0.5
        radio_landmark_loss = 1.0
    return radio_cls_loss,radio_bbox_loss,radio_landmark_loss

def cal_acc(prob_cls, gt_cls):
    '''
    计算精度：标签为0 或者1 的有多少被成功预测
    prob_cls：预测的概率 tensor [batch_size,2,1,1]--[batch_size,] 预测为face概率
    gt:gt的标签  大小为[batch_size] [0,-1,-2,0,1]
    '''
    prob_cls=prob_cls.squeeze()#[batch_size,2]
    #print(prob_cls)
    #print(gt_cls)
    mask = torch.ge(gt_cls,0)#[1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
    # print(mask)
    valid_gt_cls = torch.masked_select(gt_cls,mask)#[num,] [1,0,1,1,0]
    valid_prob_cls = torch.stack((torch.masked_select(prob_cls[:,0],mask),torch.masked_select(prob_cls[:,1],mask)),dim=-1)#[num,2] [0.1,0.5,0.6,0.4,0.8]
    size=valid_gt_cls.size()[0]
    face_index=torch.argmax(valid_prob_cls,dim=1)
    # print(valid_gt_cls)
    # print(face_index)
    sum_tensor=torch.sum(torch.eq(valid_gt_cls.float(),face_index.float()))
    acc=sum_tensor.float()/size
    # print("size:{} Tsum:{} acc:{:.4f}".format(size,sum_tensor.float(),acc))
    return acc

def compute_recall_precision(gt_cls,prob_cls):
    '''
    计算召回率： 标签为1的结果，有多少被成功预测
    准确率：一个batch中预测正确的概率
    prob_cls：预测的概率 tensor [batch_size,2,1,1]--[batch_size,] 预测为face概率 #[0.6,0.5,0.7,0.2,0.4,0.3,0.2]
    gt:gt的标签  大小为[batch_size] [0,-1,1,-2,0,1,1]
    '''
    threshold=0.5
    prob_cls=F.softmax(prob_cls,dim=1)
   
   
    #取出标签为0,1 的label 以及对应的 预测概率
    gt_cls = torch.squeeze(gt_cls)     #[batch_size,]
    prob_cls = prob_cls.squeeze()[:,1] #[batch_size,]
    # print("111",gt_cls)
    # print("111",prob_cls)
    mask = torch.ge(gt_cls,0)#[1,0,1,0,1,1,1]
    valid_gt_cls = torch.masked_select(gt_cls,mask)#[num,] [0.6,0.7,0.4,0.3,0.2]
    valid_prob_cls = torch.masked_select(prob_cls,mask)#[num,] [0,1,0,1,1]
    # print("222",valid_gt_cls)
    # print("222",valid_prob_cls)
    # print(valid_prob_cls[10:30])
    # print(valid_gt_cls[10:30])

    #预测为1的总数
    valid_prob_mask=torch.ge(valid_prob_cls,threshold)#[1,1,0,0,0]
    TPAddFP=torch.sum(valid_prob_mask).float()#2
    
    # print("333",valid_prob_mask,TPAddFP)

    #标签为1的总数
    valid_gt_mask=torch.gt(valid_gt_cls,0)#[1,1,0,0,1]
    TPAddFN=torch.sum(valid_gt_mask).float()#3
    # print("333",valid_gt_mask)
    #标签为1，并且预测正确的个数
    TP=torch.sum(torch.masked_select(valid_prob_mask,valid_gt_mask))#[1,1,0]
    # TP=torch.sum(torch.eq(valid_prob_mask,valid_gt_mask)).float()
    if TPAddFP==0:
        #print("预测为1的总数为0---",valid_prob_cls.shape,valid_gt_cls,valid_prob_cls)
        precision=-1
    else:
        precision=TP/TPAddFP

    if TPAddFN==0:
        #print("标签为1的总数为0---",valid_gt_cls)
        recall=-1
    else:
        recall=TP/TPAddFN
    
    # print("TP:",TP)
    # print("precision:",precision)
    # print("recall:",recall)
    return recall,precision

    # print("111",gt_cls)
    # print("111",prob_cls)
    # #print(prob_cls,gt_cls)
    # #过滤出标签为1的样本索引，然后取出一个batch_size中 标签为1的样本，以及该样本对应的预测概率
    # #计算召回率
    # mask = torch.gt(gt_cls,0)#[0,0,1,0,0,1]
    # valid_gt_cls = torch.masked_select(gt_cls,mask)#[num,] [1,1]
    # valid_prob_cls = torch.masked_select(prob_cls,mask)#[num,] [0.7,0.3]
    # print("222",valid_gt_cls)
    # print("222",valid_prob_cls)
    # #计算一个batch_size样本中，标签为1的样本总和
    # label_pos_size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])

    # #预测概率>=0.6的为正例，并计算预测的正例与标签为1的样本相等的mask
    # prob_ones = torch.ge(valid_prob_cls,threshold)#[1,0]
    # right_ones = torch.eq(prob_ones.float(),valid_gt_cls.float())# [1,0]torch.cuda.ByteTensor
    # #print("the num of pos {}，but {} be judged to 1".format(label_pos_size,torch.sum(right_ones).float()))
    # recall=torch.div(torch.sum(right_ones).float(),float(label_pos_size)) if label_pos_size!=0 else -1
    # if label_pos_size==0:
    #     print()
    # # print("recall:",recall)
    # #计算准确率
    # #1）过滤出标签为0 1 的样本
    # mask = torch.ge(gt_cls,0)#[1,0,1,0,1,1]
    # valid_gt_cls = torch.masked_select(gt_cls,mask)#[num,] [0,1,0,1]
    # valid_prob_cls = torch.masked_select(prob_cls,mask)#[num,] [0.6,0.7,0.4,0.3]
    # #获取所有预测为1的样本mask，并取出对应的索引
    # prob_ones = torch.ge(valid_prob_cls,threshold)#[1,1,0,0]
    # num=torch.sum(prob_ones.float())
    # rea=torch.masked_select(valid_gt_cls,prob_ones)#[0,1]   预测为1的样本对应的实际的标签
    # pre_cor_num=torch.sum(rea.float())

    # #print("{} samples be judges to 1 but {} samples judged correctly ".format(num,torch.sum(pre_cor_num).float()))
    # precision=pre_cor_num.float()/num.float() if num!=0 else -1
    # # print("acc:",acc)
    # return recall,precision

k=0.7
# cls loss 只考虑 label 1和0
# gt_label tensor([1,0,-1,-2])--batch_size
# pred_label (batch_size,2,1,1)
def cal_cls_loss(gt_label,pred_label): 
    pred_label = torch.squeeze(pred_label)#(batch_size,2)
    gt_label = torch.squeeze(gt_label)#(batch_size)--[1,0,-1,-2]
    # print("111",gt_label)
    # print("111",pred_label)

    # get the mask element which >= 0, only 0 and 1 can effect the detection loss
    mask = torch.ge(gt_label,0)#(batch_size)--[1,1,0,0]
    #mask = torch.eq(unmask,0)#[0,0,1,1]
    # chose_index
    chose_index=torch.nonzero(mask.data)#[[0,1]] shape=[2,1]
    #print(chose_index.shape)
    chose_index = torch.squeeze(chose_index)#[1,0]
    # valid_gt_label = torch.masked_select(gt_label,mask)#[num,] 
    # valid_pred_label = torch.masked_select(pred_label,mask)#[num,] 
    
    valid_gt_label=gt_label[chose_index]
    valid_pred_label=pred_label[chose_index,:]
    loss = F.cross_entropy(valid_pred_label, valid_gt_label, reduction='none')
    topk = int( k* loss.size(0))
    loss, _ = torch.topk(loss, topk)
    loss = torch.mean(loss)
    return loss

def cal_box_loss(gt_label,gt_offset,pred_offset):
        
    pred_offset = torch.squeeze(pred_offset)#去掉维数为1的维度(batch_size,4)
    gt_offset = torch.squeeze(gt_offset)# (batch_size,4)
    gt_label = torch.squeeze(gt_label)#(batch_size)--example [1,0,-1,-2]
    # print("222",gt_label)
    # print("222",gt_offset)
    # print("222",pred_offset)
    
    mask1=torch.eq(gt_label,1)#---[1,0,0,0] 找到等于1的索引
    mask2=torch.eq(gt_label,-1)#--[0,0,1,0] 找到等于-1的索引
    mask=mask1+mask2#[1,0,1,0]
    #get the mask element which != 0
    #mask=torch.ne(gt_label,0)#--[1,0,1,1] 找到不等于0的索引
    # unmask = torch.eq(gt_label,0)#other可以为Tensor或者float，判断两个是否相等，得到0 1 Tensor
    #mask = torch.eq(unmask,0)#[0,1,0,0]
    #convert mask to dim index
    chose_index = torch.nonzero(mask.data)#mask中不等于0的索引[0,2]
    chose_index = torch.squeeze(chose_index)#[0,2]
    #only valid element can effect the loss
    valid_gt_offset = gt_offset[chose_index,:]
    valid_pred_offset = pred_offset[chose_index,:]

    # print("222",valid_gt_offset)
    # print("222",valid_pred_offset)
    loss = F.mse_loss(valid_pred_offset, valid_gt_offset)
    return loss

def cal_landmark_loss(gt_label,gt_landmark,pred_landmark):
        
    pred_landmark = torch.squeeze(pred_landmark)#去掉维数为1的维度(batch_size,10)
    gt_landmark = torch.squeeze(gt_landmark) #(batch_size,10)
    gt_label = torch.squeeze(gt_label)#(batch_size)--example [1,0,-1,-2]
    # print("333",gt_label)
    # print("333",gt_landmark)
    # print("333",pred_landmark)

    mask = torch.eq(gt_label,-2)#找到等于-2的索引 [0,0,0,1]
    chose_index = torch.nonzero(mask.data)#[3]
    chose_index = torch.squeeze(chose_index)#[3]

    valid_gt_landmark = gt_landmark[chose_index, :]
    valid_pred_landmark = pred_landmark[chose_index, :]
    # loss = F.smooth_l1_loss(valid_pred_landmark, valid_gt_landmark)
    loss = F.mse_loss(valid_pred_landmark, valid_gt_landmark)
    # print("333",valid_gt_landmark)
    # print("333",valid_pred_landmark)
    return loss


val_batch=[]
# test_img_path='/media/ubuntu_data2/02_dataset/wjw/dataset/WIDER/WIDER_train/images/0--Parade/0_Parade_marchingband_1_173.jpg'
val_img_file='/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/val'

def train_PNet(use_gpu, num_epoch=30):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #获取模型
    pnet = PNet(phase="train")
    if use_gpu:
        device_ids=[int(i) for i in args.gpus.strip().split(',')]
        print("gpus:",device_ids)
        pnet = pnet.cuda(device_ids[0])
        pnet = torch.nn.DataParallel(pnet, device_ids=device_ids)
    #获取数据加载器
    label_file=os.path.join(args.train_data_dir,'imglist_12.txt')
    dataloders,dataset_size=readDataLoader(label_file,batch_size=args.batch_size,num_workers=args.num_workers)
    #1.定义优化器
    optimizer = optim.SGD(pnet.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    # optimizer = optim.SGD(pnet.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(pnet.parameters(), lr=args.lr)
    #2.定义损失函数
    cls_factor,box_factor,landmark_factor=get_loss_ratio(net_name=args.net_name)
    lossfn=LossFn(cls_factor=cls_factor, box_factor=box_factor, landmark_factor=landmark_factor)
    
    #3.定义学习率变化函数
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9) #每过1个epoch训练，学习率就乘gamma
    # optim.lr_scheduler.MultiStepLR(optimizer, step_size=1, gamma=0.1)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[12,20,25], gamma=0.1)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[6,12,18], gamma=0.1)

    # args.weight_decay=1
    # # 初始化正则化
    # if args.weight_decay>0:
    #     reg_loss=Regularization(pnet, args.weight_decay, p=2).to(device)
    # else:
    #     print("no regularization")


    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch,num_epoch+args.start_epoch-1))
        print('-'*10)
        # exp_lr_scheduler.step(epoch)
        scheduler.step()
        pnet.train(True)  # Set model to training mode
        # reset epoch loss counters
        # cls_loss=0
        # box_offset_loss = 0
        # landmark_loss = 0

        #TODO 数据加载 (输入图片矩阵,该图片中gt的标签，gt的box坐标，landmark的坐标)
        starttime=time.time()
        # t6=0
        for batch_idx, (path,inputs,gt_label,gt_bbox,gt_landmark) in enumerate(dataloders):
            
            # print("path 大小:{}".format(path))
            # print("inputs 大小:{}".format(inputs))
            # print("gt_label 大小:{}".format(gt_label))
            # print("gt_bbox 大小:{}".format(gt_bbox))
            # print("gt_landmark 大小:{}".format(gt_landmark))
            # wrap them in Variable
            if use_gpu:
                inputs = inputs.cuda()
                gt_label = gt_label.cuda()
                gt_bbox=gt_bbox.cuda()
                gt_landmark=gt_landmark.cuda()
            t2=time.time()
            dur=t2-starttime
            # print("{} it takes {}s on getting each batch_size data".format(datetime.datetime.now(),dur/(batch_idx+1)))#5.442047158877055s

            #将一个batch作为验证集看网络的分类效果
            if epoch==0 and batch_idx==0:
                print("********")
                val_batch.append(inputs)
                val_batch.append(gt_label)
                val_batch.append(path)
                # for img in val_batch[2]:
                #     print(img)
            if batch_idx%args.print_freq==0:
                index=val_batch[1].ge(0)
                index=torch.nonzero(index)
                index=torch.squeeze(index)
                val_input=val_batch[0][index,:,:,:]
                val_gt_cls=val_batch[1][index]
                # val_path=val_batch[2][index]
                val_cls_pred,val_box_pred,val_landmark_pred=pnet(val_input)
                # val_path=val_batch[2][index.tolist()]
                if batch_idx==0:
                    print(index.tolist())
                    for i in range(27):
                        shutil.copy(val_batch[2][index.tolist()[i]],val_img_file)
                        print(val_batch[2][index.tolist()[i]])
                        # os.rename()
                val_cls_pred=F.softmax(val_cls_pred,dim=1)
                # print(val_cls_pred.shape)
                # print(val_gt_cls.shape)
                print(val_cls_pred.squeeze()[0:27,1])
                print(val_gt_cls[0:27])
                # test_img=cv2.imread(test_img_path)
                # test_img=test_img[...,[2,1,0]]#hwc rgb
                # img_transforms = transforms.Compose([
                #                  transforms.ToTensor(),
                #                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                # test_img=img_transforms(test_img)
                # test_img=test_img.unsqueeze(0)#
                # print(test_img.shape)
                # img_cls_pred,_,_=pnet(test_img)
                # img_cls_pred=F.softmax(img_cls_pred,dim=1)
                # print("img:",img_cls_pred.squeeze().permute(1,2,0)[100:110,100:110,:])
        
            t3=time.time()
            cls_pred,box_pred,landmark_pred=pnet(inputs)
            t4=time.time()
            #print("t4-t3:",t4-t3)
            #cls_pred.shape (batch_size,2,1,1)
            #cls_pred=F.softmax(cls_pred,dim=1)
            
            #计算损失函数值
            loss = lossfn.cls_loss(gt_label,cls_pred)
            topk = int( k* loss.size(0))
            loss, _ = torch.topk(loss, topk)
            cls_loss = torch.mean(loss)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_pred)
            landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_pred)
            # landmark_loss=0
            # loss_sum = cls_loss+box_offset_loss+landmark_loss
            # cls_loss = cal_cls_loss(gt_label,cls_pred)
            # box_offset_loss = cal_box_loss(gt_label,gt_bbox,box_pred)
            # landmark_loss = cal_landmark_loss(gt_label,gt_landmark,landmark_pred)
            loss_sum = cls_loss * cls_factor + box_offset_loss * box_factor +landmark_loss*landmark_factor
            # print("loss_sum:",loss_sum)loss_sum: tensor(0.9309, device='cuda:0', grad_fn=<AddBackward0>)
            t5=time.time()
            #print("t5-t4:",t5-t4)#0.0019407272338867188s
            #清零优化器的梯度值
            optimizer.zero_grad()
            #损失函数反向传播
            loss_sum.backward()
            # torch.nn.utils.clip_grad_norm_(pnet.parameters(), 120.0)
            #优化器更新一次
            optimizer.step()
            t6=time.time()
            # print("t6-t2:",t6-t2)#0.003720521926879883s
            if batch_idx %args.print_freq==0:
                recall,precision=compute_recall_precision(gt_label,cls_pred)
                acc=cal_acc(cls_pred,gt_label)
                show2 = cls_loss.item()
                show3 = box_offset_loss.item()
                show4 = landmark_loss.item()
                # show4=0
                show5 = loss_sum.item()
                # print("{}: Epoch: {}, [batch:{}/{}], all_loss: {:.6f},cls_loss:{:.6f},box_offset_loss:{:.6f},landmark_loss:{:.6f},lr:{:.10f} acc:{:.5f} recall:{} pres:{}".format
                # (datetime.datetime.now(),epoch,batch_idx,round(dataset_size/args.batch_size)-1,show5,show2,show3,show4,exp_lr_scheduler.get_lr()[0],acc,recall,precision))
                print("{}: Epoch: {}, [batch:{}/{}], all_loss: {:.6f},cls_loss:{:.6f},box_offset_loss:{:.6f},landmark_loss:{:.6f},lr:{:.10f} acc:{:.5f} recall:{:.5f} pres:{:.5f}".format
                (datetime.datetime.now(),epoch,batch_idx,round(dataset_size/args.batch_size)-1,show5,show2,show3,show4,optimizer.param_groups[0]['lr'],acc,recall,precision))
                #return
                # print("{}: Epoch: {}, [batch:{}/{}],det loss: {},bbox loss: {}, landmark loss: {}, all_loss: {}, lr:{} ".format
                # (datetime.datetime.now(),epoch,batch_idx,round(dataset_size/args.batch_size)-1,show2,show3,show4,show5,exp_lr_scheduler.get_lr()[0]))
        if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
        torch.save(pnet, os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth.tar"))
        torch.save({"epoch":epoch,
                    "model_state_dict":pnet.module.state_dict(),
                    "optimizer":optimizer.state_dict()
                        }, os.path.join(args.save_path, "checkpoints_epoch_" + str(epoch) + ".tar"))
        # torch.save(pnet.state_dict(), os.path.join(args.save_path,"pnet_epoch_%d.pt" % cur_epoch))
        # torch.save(pnet, os.path.join(args.save_path,"pnet_epoch_model_%d.pkl" % cur_epoch))


#python train.py --net_name=RNet  --lr  0.001 --save_path ./Rnet_weight/ --num_epoch 22 --print_freq 100
def train_RNet(use_gpu,img_size, num_epoch=30):
    #获取模型
    rnet = RNet()
    if use_gpu:
        device_ids=[int(i) for i in args.gpus.strip().split(',')]
        print("gpus:",device_ids)
        rnet = rnet.cuda(device_ids[0])
        rnet = torch.nn.DataParallel(rnet, device_ids=device_ids)
    #获取数据加载器
    pos_file =args.data_dir+"{}/pos_{}.txt".format(img_size,img_size)
    neg_file =args.data_dir+"{}/neg_{}.txt".format(img_size,img_size)
    part_file =args.data_dir+"{}/part_{}.txt".format(img_size,img_size)
    land_file =args.data_dir+"{}/landmark_{}_aug.txt".format(img_size,img_size) 

    # posdata,pos_size=readDataLoader(label_file=pos_file,batch_size=64,num_workers=30)
    # negdata,neg_size=readDataLoader(label_file=neg_file,batch_size=192,num_workers=30)
    # partdata,part_size=readDataLoader(label_file=part_file,batch_size=64,num_workers=30)
    # landdata,land_size=readDataLoader(label_file=land_file,batch_size=64,num_workers=30)
    posdata,pos_size=readDataLoader(label_file=pos_file,batch_size=256,num_workers=30)
    negdata,neg_size=readDataLoader(label_file=neg_file,batch_size=768,num_workers=30)
    partdata,part_size=readDataLoader(label_file=part_file,batch_size=256,num_workers=30)
    landdata,land_size=readDataLoader(label_file=land_file,batch_size=256,num_workers=30)
    pos_iter=iter(posdata)
    neg_iter=iter(negdata)
    part_iter=iter(partdata)
    land_iter=iter(landdata)
    #1.定义优化器
    optimizer = optim.SGD(rnet.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    #2.定义损失函数
    cls_factor,box_factor,landmark_factor=get_loss_ratio(net_name=args.net_name)
    lossfn=LossFn(cls_factor=cls_factor, box_factor=box_factor, landmark_factor=landmark_factor)
    
    #120000个pos 一个batch 64个  一轮需要1875个step 
    step_each_epoch=468#1875
    max_step=num_epoch*step_each_epoch
    acc=0
    print("start trainning!!! the max_step:",max_step)
    # Set model to training mode
    rnet.train(True)
    starttime=time.time()
    path=[]
    #TODO 3.定义学习率变化函数 确定迭代次数为多少时需要改变学习率
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[6*step_each_epoch,12*step_each_epoch,18*step_each_epoch], gamma=0.1)

    for step in range(max_step):
        scheduler.step()
        #TODO 数据加载 (输入图片矩阵,该图片中gt的标签，gt的box坐标，landmark的坐标)
        try:
            pos_img_path,pos_img,pos_label,pos_gt_bbox,pos_gt_landmark = next(pos_iter)
        except StopIteration:
            pos_iter = iter(posdata)
            pos_img_path,pos_img,pos_label,pos_gt_bbox,pos_gt_landmark = next(pos_iter)
        
        try:
            neg_img_path,neg_img,neg_label,neg_gt_bbox,neg_gt_landmark = next(neg_iter)
        except StopIteration:
            neg_iter = iter(negdata)
            neg_img_path,neg_img,neg_label,neg_gt_bbox,neg_gt_landmark = next(neg_iter)

        try:
            part_img_path,part_img,part_label,part_gt_bbox,part_gt_landmark = next(part_iter)
        except StopIteration:
            part_iter = iter(partdata)
            part_img_path,part_img,part_label,part_gt_bbox,part_gt_landmark = next(part_iter)

        try:
            land_img_path,land_img,land_label,land_gt_bbox,land_gt_landmark = next(land_iter)
        except StopIteration:
            land_iter = iter(landdata)
            land_img_path,land_img,land_label,land_gt_bbox,land_gt_landmark = next(land_iter)

        #从各个batch中组成训练用的mini_batch
        # path.append()
        inputs=torch.cat((pos_img,neg_img,part_img,land_img),dim=0)
        gt_label=torch.cat((pos_label,neg_label,part_label,land_label),dim=0)
        gt_bbox=torch.cat((pos_gt_bbox,neg_gt_bbox,part_gt_bbox,land_gt_bbox),dim=0)
        gt_landmark=torch.cat((pos_gt_landmark,neg_gt_landmark,part_gt_landmark,land_gt_landmark),dim=0)

        # print("path 大小:{}".format(path))
        
        # print("inputs 大小:{}".format(inputs.shape))
        # print("gt_label 大小:{}".format(gt_label.shape))
        # print("gt_bbox 大小:{}".format(gt_bbox.shape))
        # print("gt_landmark 大小:{}".format(gt_landmark.shape))
        # wrap them in Variable
        if use_gpu:
            inputs = inputs.cuda()
            gt_label = gt_label.cuda()
            gt_bbox=gt_bbox.cuda()
            gt_landmark=gt_landmark.cuda()
        t2=time.time()
        dur=t2-starttime
        # print("{} it takes {}s on getting each batch_size data".format(datetime.datetime.now(),dur/(step+1)))#3.442047158877055s

        #将一个batch作为验证集看网络的分类效果
        if step==0:
            print("********")
            val_batch.append(inputs)
            val_batch.append(gt_label)
            # val_batch.append(path)
            # for img in val_batch[2]:
            #     print(img)
        if step % args.print_freq==0:
            index=val_batch[1].ge(0)
            index=torch.nonzero(index)
            index=torch.squeeze(index)
            val_input=val_batch[0][index,:,:,:]
            val_gt_cls=val_batch[1][index]
            # val_path=val_batch[2][index]
            val_cls_pred,val_box_pred,val_landmark_pred=rnet(val_input)
            # val_path=val_batch[2][index.tolist()]
            # if batch_idx==0:
            #     print(index.tolist())
            #     for i in range(27):
            #         shutil.copy(val_batch[2][index.tolist()[i]],val_img_file)
            #         print(val_batch[2][index.tolist()[i]])
                    # os.rename()
            val_cls_pred=F.softmax(val_cls_pred,dim=1)
            # print(val_cls_pred.shape)
            # print(val_gt_cls.shape)
            print(val_cls_pred.squeeze()[46:82,1])
            print(val_gt_cls[46:82])

    
        t3=time.time()
        cls_pred,box_pred,landmark_pred=rnet(inputs)#(batch_size,2) (batch_size,4) (batch_size,10)
        t4=time.time()

        #计算损失函数值
        loss = lossfn.cls_loss(gt_label,cls_pred)
        topk = int( k* loss.size(0))
        loss, _ = torch.topk(loss, topk)
        cls_loss = torch.mean(loss)
        box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_pred)
        landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_pred)
        loss_sum = cls_loss * cls_factor + box_offset_loss * box_factor +landmark_loss*landmark_factor
        # print("loss_sum:",loss_sum)loss_sum: tensor(0.9309, device='cuda:0', grad_fn=<AddBackward0>)
        t5=time.time()
        #print("t5-t4:",t5-t4)#0.0019407272338867188s
        #清零优化器的梯度值
        optimizer.zero_grad()
        #损失函数反向传播
        loss_sum.backward()
        #优化器更新一次
        optimizer.step()
        t6=time.time()
        # print("t6-t2:",t6-t2)#0.003720521926879883s
        if step %args.print_freq==0:
            recall,precision=compute_recall_precision(gt_label,cls_pred)
            acc=cal_acc(cls_pred,gt_label)
            show2 = cls_loss.item()
            show3 = box_offset_loss.item()
            show4 = landmark_loss.item()
            show5 = loss_sum.item()
            # print("{}: Epoch: {}, [batch:{}/{}], all_loss: {:.6f},cls_loss:{:.6f},box_offset_loss:{:.6f},landmark_loss:{:.6f},lr:{:.10f} acc:{:.5f} recall:{} pres:{}".format
            # (datetime.datetime.now(),epoch,batch_idx,round(dataset_size/args.batch_size)-1,show5,show2,show3,show4,exp_lr_scheduler.get_lr()[0],acc,recall,precision))
            print("{}:  [Step:{}/{}], all_loss: {:.6f},cls_loss:{:.6f},box_offset_loss:{:.6f},landmark_loss:{:.6f},lr:{:.10f} acc:{:.5f} recall:{:.5f} pres:{:.5f}".format
            (datetime.datetime.now(),max_step,step,show5,show2,show3,show4,optimizer.param_groups[0]['lr'],acc,recall,precision))
            #return
            # print("{}: Epoch: {}, [batch:{}/{}],det loss: {},bbox loss: {}, landmark loss: {}, all_loss: {}, lr:{} ".format
            # (datetime.datetime.now(),epoch,batch_idx,round(dataset_size/args.batch_size)-1,show2,show3,show4,show5,exp_lr_scheduler.get_lr()[0]))
        if (step+1)%step_each_epoch==0:
            if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
            torch.save(rnet, os.path.join(args.save_path, "iter_" + str(step) + ".pth.tar"))
            torch.save({"epoch":step,
                        "acc":acc,
                        "model_state_dict":rnet.module.state_dict(),
                        "optimizer":optimizer.state_dict()
                            }, os.path.join(args.save_path, "checkpoints_iter_" + str(step) + ".tar"))


def train_ONet(use_gpu,img_size, num_epoch=30):
    #获取模型
    onet = ONet()
    if use_gpu:
        device_ids=[int(i) for i in args.gpus.strip().split(',')]
        print("gpus:",device_ids)
        onet = onet.cuda(device_ids[0])
        onet = torch.nn.DataParallel(onet, device_ids=device_ids)
    #获取数据加载器
    pos_file =args.data_dir+"{}/pos_{}.txt".format(img_size,img_size)
    neg_file =args.data_dir+"{}/neg_{}.txt".format(img_size,img_size)
    part_file =args.data_dir+"{}/part_{}.txt".format(img_size,img_size)
    land_file =args.data_dir+"{}/landmark_{}_aug.txt".format(img_size,img_size) 

    # posdata,pos_size=readDataLoader(label_file=pos_file,batch_size=64,num_workers=20)
    # negdata,neg_size=readDataLoader(label_file=neg_file,batch_size=192,num_workers=20)
    # partdata,part_size=readDataLoader(label_file=part_file,batch_size=64,num_workers=20)
    # landdata,land_size=readDataLoader(label_file=land_file,batch_size=64,num_workers=20)

    posdata,pos_size=readDataLoader(label_file=pos_file,batch_size=256,num_workers=20)
    negdata,neg_size=readDataLoader(label_file=neg_file,batch_size=768,num_workers=20)
    partdata,part_size=readDataLoader(label_file=part_file,batch_size=256,num_workers=20)
    landdata,land_size=readDataLoader(label_file=land_file,batch_size=256,num_workers=20)
    pos_iter=iter(posdata)
    neg_iter=iter(negdata)
    part_iter=iter(partdata)
    land_iter=iter(landdata)
    #1.定义优化器
    optimizer = optim.SGD(onet.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    #2.定义损失函数
    cls_factor,box_factor,landmark_factor=get_loss_ratio(net_name=args.net_name)
    lossfn=LossFn(cls_factor=cls_factor, box_factor=box_factor, landmark_factor=landmark_factor)
    
    #120000个pos 一个batch 64个  一轮需要1875个step 
    step_each_epoch=468#1875
    max_step=num_epoch*step_each_epoch
    acc=0
    print("start trainning!!! the max_step:",max_step)
    # Set model to training mode
    onet.train(True)
    starttime=time.time()
    path=[]
    #TODO 3.定义学习率变化函数 确定迭代次数为多少时需要改变学习率
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[6*step_each_epoch,12*step_each_epoch,18*step_each_epoch], gamma=0.1)

    for step in range(max_step):
        scheduler.step()
        #TODO 数据加载 (输入图片矩阵,该图片中gt的标签，gt的box坐标，landmark的坐标)
        try:
            pos_img_path,pos_img,pos_label,pos_gt_bbox,pos_gt_landmark = next(pos_iter)
        except StopIteration:
            pos_iter = iter(posdata)
            pos_img_path,pos_img,pos_label,pos_gt_bbox,pos_gt_landmark = next(pos_iter)
        
        try:
            neg_img_path,neg_img,neg_label,neg_gt_bbox,neg_gt_landmark = next(neg_iter)
        except StopIteration:
            neg_iter = iter(negdata)
            neg_img_path,neg_img,neg_label,neg_gt_bbox,neg_gt_landmark = next(neg_iter)

        try:
            part_img_path,part_img,part_label,part_gt_bbox,part_gt_landmark = next(part_iter)
        except StopIteration:
            part_iter = iter(partdata)
            part_img_path,part_img,part_label,part_gt_bbox,part_gt_landmark = next(part_iter)

        try:
            land_img_path,land_img,land_label,land_gt_bbox,land_gt_landmark = next(land_iter)
        except StopIteration:
            land_iter = iter(landdata)
            land_img_path,land_img,land_label,land_gt_bbox,land_gt_landmark = next(land_iter)

        #从各个batch中组成训练用的mini_batch
        # path.append()
        inputs=torch.cat((pos_img,neg_img,part_img,land_img),dim=0)
        gt_label=torch.cat((pos_label,neg_label,part_label,land_label),dim=0)
        gt_bbox=torch.cat((pos_gt_bbox,neg_gt_bbox,part_gt_bbox,land_gt_bbox),dim=0)
        gt_landmark=torch.cat((pos_gt_landmark,neg_gt_landmark,part_gt_landmark,land_gt_landmark),dim=0)

        # print("path 大小:{}".format(path))
        
        # print("inputs 大小:{}".format(inputs.shape))
        # print("gt_label 大小:{}".format(gt_label.shape))
        # print("gt_bbox 大小:{}".format(gt_bbox.shape))
        # print("gt_landmark 大小:{}".format(gt_landmark.shape))
        # wrap them in Variable
        if use_gpu:
            inputs = inputs.cuda()
            gt_label = gt_label.cuda()
            gt_bbox=gt_bbox.cuda()
            gt_landmark=gt_landmark.cuda()
        t2=time.time()
        dur=t2-starttime
        # print("{} it takes {}s on getting each batch_size data".format(datetime.datetime.now(),dur/(step+1)))#3.442047158877055s

        #将一个batch作为验证集看网络的分类效果
        if step==0:
            print("********")
            val_batch.append(inputs)
            val_batch.append(gt_label)
            # val_batch.append(path)
            # for img in val_batch[2]:
            #     print(img)
        if step % args.print_freq==0:
            index=val_batch[1].ge(0)
            index=torch.nonzero(index)
            index=torch.squeeze(index)
            val_input=val_batch[0][index,:,:,:]
            val_gt_cls=val_batch[1][index]
            # val_path=val_batch[2][index]
            val_cls_pred,val_box_pred,val_landmark_pred=onet(val_input)
            # val_path=val_batch[2][index.tolist()]
            # if batch_idx==0:
            #     print(index.tolist())
            #     for i in range(27):
            #         shutil.copy(val_batch[2][index.tolist()[i]],val_img_file)
            #         print(val_batch[2][index.tolist()[i]])
                    # os.rename()
            val_cls_pred=F.softmax(val_cls_pred,dim=1)
            # print(val_cls_pred.shape)
            # print(val_gt_cls.shape)
            print(val_cls_pred.squeeze()[46:82,1])
            print(val_gt_cls[46:82])

    
        t3=time.time()
        cls_pred,box_pred,landmark_pred=onet(inputs)#(batch_size,2) (batch_size,4) (batch_size,10)
        t4=time.time()

        #计算损失函数值
        loss = lossfn.cls_loss(gt_label,cls_pred)
        topk = int( k* loss.size(0))
        loss, _ = torch.topk(loss, topk)
        cls_loss = torch.mean(loss)
        box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_pred)
        landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_pred)
        loss_sum = cls_loss * cls_factor + box_offset_loss * box_factor +landmark_loss*landmark_factor
        # print("loss_sum:",loss_sum)loss_sum: tensor(0.9309, device='cuda:0', grad_fn=<AddBackward0>)
        t5=time.time()
        #print("t5-t4:",t5-t4)#0.0019407272338867188s
        #清零优化器的梯度值
        optimizer.zero_grad()
        #损失函数反向传播
        loss_sum.backward()
        #优化器更新一次
        optimizer.step()
        t6=time.time()
        # print("t6-t2:",t6-t2)#0.003720521926879883s
        if step %args.print_freq==0:
            recall,precision=compute_recall_precision(gt_label,cls_pred)
            acc=cal_acc(cls_pred,gt_label)
            show2 = cls_loss.item()
            show3 = box_offset_loss.item()
            show4 = landmark_loss.item()
            show5 = loss_sum.item()
            # print("{}: Epoch: {}, [batch:{}/{}], all_loss: {:.6f},cls_loss:{:.6f},box_offset_loss:{:.6f},landmark_loss:{:.6f},lr:{:.10f} acc:{:.5f} recall:{} pres:{}".format
            # (datetime.datetime.now(),epoch,batch_idx,round(dataset_size/args.batch_size)-1,show5,show2,show3,show4,exp_lr_scheduler.get_lr()[0],acc,recall,precision))
            print("{}:  [Step:{}/{}], all_loss: {:.6f},cls_loss:{:.6f},box_offset_loss:{:.6f},landmark_loss:{:.6f},lr:{:.10f} acc:{:.5f} recall:{:.5f} pres:{:.5f}".format
            (datetime.datetime.now(),max_step,step,show5,show2,show3,show4,optimizer.param_groups[0]['lr'],acc,recall,precision))
            #return
            # print("{}: Epoch: {}, [batch:{}/{}],det loss: {},bbox loss: {}, landmark loss: {}, all_loss: {}, lr:{} ".format
            # (datetime.datetime.now(),epoch,batch_idx,round(dataset_size/args.batch_size)-1,show2,show3,show4,show5,exp_lr_scheduler.get_lr()[0]))
        if (step+1)%step_each_epoch==0:
            if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
            torch.save(onet, os.path.join(args.save_path, "iter_" + str(step) + ".pth.tar"))
            torch.save({"epoch":step,
                        "acc":acc,
                        "model_state_dict":onet.module.state_dict(),
                        "optimizer":optimizer.state_dict()
                            }, os.path.join(args.save_path, "checkpoints_iter_" + str(step) + ".tar"))

if __name__ == '__main__':
    
    if  args.net_name =="PNet":
        train_PNet(use_gpu=True,num_epoch=args.num_epochs)
    if  args.net_name =="RNet":
        train_RNet(use_gpu=True,num_epoch=args.num_epochs,img_size=24)
    if  args.net_name =="ONet":
        train_ONet(use_gpu=True,num_epoch=args.num_epochs,img_size=48)
