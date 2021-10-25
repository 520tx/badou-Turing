import torch 
import torch.nn as nn
import numpy as np
import time
from torchvision import transforms 
import cv2
import os
import sys
sys.path.insert(0,"..")
from prepare_data.utils import *
import torch.nn.functional as F
import datetime
class MtcnnDetector(object):
    def __init__(self,pnet,rnet,onet,min_face_size=20,stride=2,threshold=[0.6, 0.7, 0.7],scale_factor=0.79, slide_window=False,use_gpu=True):
        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor
        self.slide_window = slide_window
        self.use_gpu = use_gpu
    
    def convert_to_square(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox
        Returns:
        -------
            square bbox
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox

    
    def calibrate_box(self, bbox, reg):
        """
            calibrate bboxes
        Parameters:
        ----------
            bbox: numpy array, shape n x 5 传入rnet网络的图片box 在原图中的坐标 相当于nx1,ny1,nx2,ny2
                input bboxes
            reg:  numpy array, shape n x 4  rnet预测的box的offset 用于还原GT (offset_x1*size)+nx1 ...
                bboxes adjustment
        Returns:
        -------
            bboxes after refinement
        """

        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c


    def generate_bbox(self,cls_pred,box_pred,scale, threshold):
        #TODO 确保cls_pred得到的是softmax概率 第一次PNet概率输出没有加上softmax
        '''
        cls_pred: tensor(1,2,H,W) detect score for each position
        box_pred:     tensor(1,4,H,W)  bbox
        '''
        stride = 2
        cellsize = 12
        #向量的处理
        #(1,2,H,W)-(1,2,H,W)softmax-(2,H,W)--(H,W,2)--(H,W)
        face_pred=cls_pred[0,...].permute(1,2,0)[...,1]
        # face_pred=cls_pred[0,...].permute(1,2,0)           #(1,2,H,W)--(2,H,W)--(H,W,2)
        # face_pred=F.softmax(face_pred,dim=2)[...,1]       
        box_pred=box_pred[0,...].permute(1,2,0)#(1,4,H,W)--(4,H,W)--(H,W,4)
        #获取满足要求的mask
        index=torch.ge(face_pred,threshold)
        #获取mask,所在的坐标，相当于feature map上满足要求的坐标值，用于转换到原图  torch.LongTensor-->torch.FloatTensor
        fm_index=torch.nonzero(index).float()
        if fm_index.shape[0] == 0:#[num,2] 
            return torch.tensor([])
        #获取满足要求的boundingbox的预测的偏移坐标,以及预测结果
        face_pred=face_pred[index]#[num,]
        box_pred_keep=box_pred[index]#[num,4]
        # print("111")
        #获取满足要求的feature map像素坐标 在原图的左上角坐标 x1,y1
        ori_x1y1=torch.round((fm_index*stride)/scale)#[num,2]
        #获取满足要求的feature map像素坐标 在原图的右下角坐标 x2,y2
        ori_x2y2=torch.round((fm_index*stride+cellsize)/scale)
        #获取满足要求的预测概率
        face_pred=face_pred.unsqueeze(0).reshape(-1,1)# #[num,]--[1,num]--[num,1]
        res=torch.cat((ori_x1y1,ori_x2y2,face_pred,box_pred_keep),dim=1)#[num,9]
        return res.cpu().detach().numpy()

    def np_generate_bbox(self, cls_map, reg, scale, threshold):
        """
            generate bbox from feature cls_map according to the threshold
        Parameters:
        ----------
            cls_map: tensor(1,2,H,W) detect score for each position
            reg:     tensor(1,4,H,W)  bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array num*9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)
        """
        cls_map_view=cls_map[0,:,:,:]#tensor(2,H,W)
        # print(cls_map_view[:,0:10,0:10])
        cls_map_view=cls_map_view[1,:,:]#tensor(H,W)
        # print(cls_map_view[0:10,0:10])
        cls_map_view=cls_map_view.cpu().detach().numpy() #numpy (H W)
        # print("cls_map_view size:",cls_map_view.shape)#cls_map_view size: (99, 140)
        # index of class_prob larger than threshold
        t_index = np.where(cls_map_view > threshold)#p:0.6 t_index为 numpy 
        #t_index[0]为所有cls_map_view中所有满足要求的H索引
        #t_index[1]为所有cls_map_view中所有满足要求的W索引

        reg=reg[0,:,:,:]##tensor(4,H,W)
        reg=reg.cpu().detach().numpy()#numpy (4,H,W)
        reg=reg.transpose((1,2,0))#numpy (H,W,4)
        #print("reg size:",reg.shape) (99, 140, 4)
        # reg_np=reg_np.swapaxes(1,0)#numpy (H,4,W)
        # reg_np=reg_np.swapaxes(2,1)#numpy (H,W,4)
        # 所有的特征图的阈值都不满足
        if t_index[0].size == 0:
            return np.array([])
        
        # offset
        stride = 2
        cellsize = 12
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]
        reg_np = np.array([dx1, dy1, dx2, dy2])

        score = cls_map_view[t_index[0], t_index[1]]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),#原图boundingbox nx1   boundingbox特征图一个点对应原图的区域
                                 np.round((stride * t_index[0]) / scale),#原图boundingbox ny1
                                 np.round((stride * t_index[1] + cellsize) / scale),#原图boundingbox nx2
                                 np.round((stride * t_index[0] + cellsize) / scale),#原图boundingbox ny2
                                 score,#原图boundingbox 预测的得分
                                 reg_np])#原图boundingbox 预测的偏移 （x1-nx1）/size  其中size=（nx2-nx1,ny2-ny1)
        #print("size,",boundingbox.shape)#(9, 91689)
        return boundingbox.T#(91689,9)

    # pre-process images
    # img_resized numpy array (h,w,c)bgr
    # img_tensor tensor (1,c,h,w) rgb
    def processed_image(self, img, scale):
        # print(img.shape,scale)
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image hwc bgr

        

        data_transforms =transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # print(img_resized[0,0,:])
        # img_tensor=img_resized
        img_tensor=img_resized[:,:,[2,1,0]]#hwc rgb
        # print(img_tensor[0,0,:])
        img_tensor=data_transforms(img_tensor)#获取该图片的pytorch tensor [3,H,W]
        img_tensor=img_tensor.unsqueeze(0)#[1,3,H,W]
        # print(img_tensor[0,:,0,0])
        return img_resized,img_tensor

    #从pnet 的网络得到的 boxes进行约束后将其进行转换得到tensor
    def rnet_processed_image(self, img):
        ret=True
        try:
            data_transforms =transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            size=(24,24)
            img_resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)  # resized image hwc bgr

            img_tensor=img_resized[:,:,[2,1,0]]#hwc rgb
            img_tensor=data_transforms(img_tensor)#获取该图片的pytorch tensor [3,24,24] chw rgb
            img_tensor=img_tensor.unsqueeze(0)#[1,3,24,24] chw rgb
        except :
            ret=False
            img_tensor=torch.zeros([])
        # print(img_tensor[0,:,0,0])
        return ret,img_tensor

    def onet_processed_image(self, img):
        ret=True
        try:
            data_transforms =transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            size=(48,48)
            img_resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)  # resized image hwc bgr

            img_tensor=img_resized[:,:,[2,1,0]]#hwc rgb
            img_tensor=data_transforms(img_tensor)#获取该图片的pytorch tensor [3,24,24] chw rgb
            img_tensor=img_tensor.unsqueeze(0)#[1,3,24,24] chw rgb
        except :
            ret=False
            img_tensor=torch.zeros([])
        # print(img_tensor[0,:,0,0])
        return ret,img_tensor

    #
    def pad(self, bboxes, w, h):
        """
            pad the the bboxes, also restrict the size of it
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ey, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        #tmpw, tmph box的宽和高
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))

        #edx, edy box的 w-1 和 h-1
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1

        # x, y, ex, ey box的x1 y1 x2 y2
        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        #如果boxes的x2->ex 超过了 原图的右下角坐标的x2->w-1  则boxes的宽为原图的右下角x2->w-1 减去 box的左上角的x1
        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        #edx[tmp_index] = (w-1)+ ((tmpw[tmp_index]-1)-ex[tmp_index])
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array [h,w,3]
            input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        net_size = 12
        self.pnet_detector.eval()
        current_scale = float(net_size) / self.min_face_size  # find initial scale 0.6
        # print("current_scale", net_size, self.min_face_size, current_scale)
        im_resized,img_tensor = self.processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape
        all_boxes = list()
        # print("开始图像金字塔！！")
        while min(current_height, current_width) > 12:#图像金字塔
            t1=time.time()
            
            # return the result predicted by pnetim_resized
            # cls_pred : H*w*2  (1,3,12,12)-->(1,2,1,1)  (1,im_resized3,current_height,current_width)-->(1,2,current_height-12,current_width-12) stride=1
            # box_pred: H*w*4   (1,3,12,12)-->(1,4,1,1)  (1,im_resized3,current_height,current_width)-->(1,4,current_height-12,current_width-12)
            # class_prob and bbox_pred
            if self.use_gpu:
                img_tensor=img_tensor.cuda()
            cls_pred, box_pred,_ = self.pnet_detector(img_tensor.cuda())
            # print(cls_pred[0,...].permute(1,2,0)[0:2,0:2,:])
            cls_pred=F.softmax(cls_pred,dim=1)
            # print(cls_pred[0,...].permute(1,2,0)[0:2,0:2,:])
            t_net=time.time()
            #print("t_net-t1:",t_net-t1)
            # boxes: tensor[num,9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)]
            # print("current_scale:",current_scale,"current_height:",current_height,"current_width:",current_width)
            boxes = self.np_generate_bbox(cls_pred, box_pred, current_scale,self.thresh[0])
            t2=time.time()

   
            #print("t2-t_net:",t2-t_net)
            # scale_factor is 0.79 in default
            current_scale *= self.scale_factor
            im_resized,img_tensor= self.processed_image(im, current_scale)#big bug im而不是im_resized
            current_height, current_width, _ = im_resized.shape
            t3=time.time()
            #print("t3-t2:",t3-t2)
            if boxes.shape[0] == 0:
                continue
            # get the index from non-maximum s
            # keep = torch_nms(torch.from_numpy(boxes[:, :4]), torch.from_numpy(boxes[:, 4]),0.5)
            # print("PNet {} boxes before nms 0.5".format(boxes.shape[0]))
            keep = py_nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]

            ###########################
            # 验证feature_map对应原图是否正确
            # im_show=im.copy()
            # cv2.imshow("mtcnn",im_show)
            # print("PNet {} boxes after nms 0.5",boxes.shape[0])
            # # print(boxes[0,1])
            # cv2.waitKey(10)
            # for i in range(boxes.shape[0]):
            #     x1=int(max(boxes[i,0],0))
            #     y1=int(max(boxes[i,1],0))
            #     x2=int(min(boxes[i,2],im_show.shape[1]))
            #     y2=int(min(boxes[i,3],im_show.shape[0]))
            #     cv2.rectangle(im_show,(x1,y1),(x2,y2),(255,0,0),1)
            #     cv2.imshow("mtcnn",im_show)
            #     if cv2.waitKey(1) == ord('q'):
            #             break
            # print("draw finish")
            # cv2.waitKey(100)
            ###########################

            all_boxes.append(boxes)
            t4=time.time()
            # print("t4-t3:{}".format(t4-t3))
        if len(all_boxes) == 0:
            return None, None, None

        all_boxes = np.vstack(all_boxes)
        
       
        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        # print("PNet {} boxes after nms 0.7".format(all_boxes.shape[0]))

        boxes = all_boxes[:, :5]
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # refine the boxes  根据预测的偏移与boundingbox求GT
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return boxes, boxes_c, None

    #rnet 网络检测 图片经过pnet后 得到了预测为人脸的boxes坐标[num,5(x1,y1,x2,y2,score)] 
    #根据boxes:将boxes转换为方形，从原图中截取相应的图片进行resize(24,24)放进训练好的rnet网络中进行预测，最终得到满足要求的boxes[num,5]
    def detect_rnet(self,im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        # cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
        cropped_ims = torch.zeros((num_boxes, 3, 24, 24), dtype=torch.float32)
        #根据boxes从原图中截取图片并且resize成24
        print("num_boxes:",num_boxes)
        for i in range(num_boxes):
            #过滤掉最小人脸
            #if tmph[i]<self.min_face_size or tmpw[i]<self.min_face_size or x[i] < 0 or y[i] < 0 or ex[i] > w - 1 or ey[i] > h - 1:
            if tmph[i]<5 or tmpw[i]<5:
                # print("error {}".format(imgpath))
                print("error {} {} {}".format(tmph[i],tmpw[i],dets[i]))
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            if tmp.shape[0]<=0 or tmp.shape[1]<=0:
                continue
            ret,rnet_tensor=self.rnet_processed_image(tmp)
            if ret==False:
                # print("throw:",imgpath)
                print("throw:",im.shape,tmp.shape)
                continue
            cropped_ims[i, :, :, :] = rnet_tensor 
        print("cropped_ims shape:",cropped_ims.shape)
        #cropped_ims [num 3 24 24]
        # cls_scores : num_data*2
        # reg: num_data*4
        # landmark: num_data*10
        #cls_scores [num,2]  reg [num,4]
        if self.use_gpu:
            cropped_ims=cropped_ims.cuda()
        cls_scores, reg, _ = self.rnet_detector(cropped_ims)
        cls_scores=F.softmax(cls_scores,dim=1)
        
        # torch to np array
        cls_scores=cls_scores.cpu().detach().numpy()
        reg=reg.cpu().detach().numpy()
        print("rnet cls_scores shape:",cls_scores.shape)
        cls_scores = cls_scores[:, 1]#np [num,]
        keep_inds = np.where(cls_scores > self.thresh[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            # landmark = landmark[keep_inds]
        else:
            return None, None, None

        keep = py_nms(boxes, 0.6)
        boxes = boxes[keep]
        boxes_c = self.calibrate_box(boxes, reg[keep])
        return boxes, boxes_c, None

    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = torch.zeros((num_boxes, 3, 48, 48), dtype=torch.float32)
        for i in range(num_boxes):
            if tmph[i]<5 or tmpw[i]<5:
                # print("error {}".format(imgpath))
                print("error {} {} {}".format(tmph[i],tmpw[i],dets[i]))
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            if tmp.shape[0]<=0 or tmp.shape[1]<=0:
                continue
            ret,onet_tensor=self.onet_processed_image(tmp)
            if ret==False:
                # print("throw:",imgpath)
                print("throw:",im.shape,tmp.shape)
                continue
            cropped_ims[i, :, :, :] = onet_tensor 

        if self.use_gpu:
            cropped_ims=cropped_ims.cuda()
        cls_scores, reg, landmark = self.onet_detector(cropped_ims)
        cls_scores=F.softmax(cls_scores,dim=1)
        cls_scores=cls_scores.cpu().detach().numpy()
        reg=reg.cpu().detach().numpy()
        landmark=landmark.cpu().detach().numpy()
        # prob belongs to face
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[2])[0]
        if len(keep_inds) > 0:
            # pickout filtered box
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None, None

        # width
        w = boxes[:, 2] - boxes[:, 0] + 1
        # height
        h = boxes[:, 3] - boxes[:, 1] + 1
        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = self.calibrate_box(boxes, reg)

        boxes = boxes[py_nms(boxes, 0.6, "Minimum")]
        keep = py_nms(boxes_c, 0.6, "Minimum")
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes, boxes_c, landmark

    # use for video
    def detect(self, img):
        """Detect face over image
        """
        boxes = None
        t = time.time()

        # pnet
        t1 = 0
        if self.pnet_detector:
            boxes, boxes_c, _ = self.detect_pnet(img)
            if boxes_c is None:
                return np.array([]), np.array([])
            t1 = time.time() - t
            t = time.time()
        print("pnet boxes shape:",boxes_c.shape)
        # rnet
        t2 = 0
        if self.rnet_detector:
            boxes, boxes_c, _ = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])

            t2 = time.time() - t
            t = time.time()

        # onet
        t3 = 0
        if self.onet_detector:
            boxes, boxes_c, landmark = self.detect_onet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])

            t3 = time.time() - t
            t = time.time()
            # print(
            #    "time cost " + '{:.3f}'.format(t1 + t2 + t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2,
            #                                                                                                  t3))

        return boxes_c, landmark

    #test_data 数据读取器 (1,h,w,3)--tensor 需要转化成numpy opencv的数据格式  可传入网络 batch_size==1
    #test_size 数据总的长度
    #all_boxes [ ] 长度为wider图片数据集的个数,每个元素为给图片对应的预测box [num,5] 5:x1,y1,x2,y2,score
    def detect_face(self, test_data,test_size):
        all_boxes = []  # save each image's bboxes
        landmarks = []

        sum_time = 0
        t1_sum = 0
        t2_sum = 0
        t3_sum = 0

        empty_array = np.array([])
        s_time = time.time()
        print_freq=100
        for batch_idx,img in enumerate(test_data):
            # print(imgpath)
            if (batch_idx) % print_freq == 0:
                c_time = (time.time() - s_time )/print_freq
                print("{} {}/{} images done".format(datetime.datetime.now(),batch_idx ,test_size))
                print('%f seconds for each image' % c_time)
                s_time = time.time()
            # print(databatch.shape)
            # print(databatch[0,0,:])
            # sys.eixt()
            # img=databatch.squeeze().numpy()#获得numpy 图片数组 h w c bgr
            # img=cv2.imread(imgpath)#获得numpy 图片数组 h w c bgr
            # print(img.shape)
            # print(img[0,0,:])
            #databatch图片送入pnet网络
            if self.pnet_detector:
                st = time.time()
                # ignore landmark boxes_c 每张图片所有的阈值>0.6 的box shape=(num,5) x1,y1,x2,y2,score
                boxes, boxes_c, landmark = self.detect_pnet(img)
                t1 = time.time() - st
                sum_time += t1
                t1_sum += t1
            if boxes_c is None:
                print("boxes_c is None...")
                all_boxes.append(empty_array)
                # pay attention
                landmarks.append(empty_array)
                continue
                #print(all_boxes)
            # rnet
            if self.rnet_detector is not None:
                t = time.time()
                boxes, boxes_c, landmark = self.detect_rnet(img, boxes_c)
                t2 = time.time() - t
                sum_time += t2
                t2_sum += t2
                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)
                    continue
            
            # onet
            if self.onet_detector:
                t = time.time()
                boxes, boxes_c, landmark = self.detect_onet(databatch, boxes_c)
                t3 = time.time() - t
                sum_time += t3
                t3_sum += t3
                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)
                    continue
            # print("{} PNet gen {} boxes".format(imgpath,boxes_c.shape[0]))
            all_boxes.append(boxes_c)
            landmark = [1]
            landmarks.append(landmark)
        print('num of images', test_size)
        print("time cost in average" +
            '{:.3f}'.format(sum_time/test_size) +
            '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1_sum/test_size, t2_sum/test_size,t3_sum/test_size))
        # num_of_data*9,num_of_data*10
        print('boxes length:',len(all_boxes))
        return all_boxes, landmarks

    def detect_single_image(self, im):
        all_boxes = []  # save each image's bboxes

        landmarks = []

       # sum_time = 0

        t1 = 0
        if self.pnet_detector:
          #  t = time.time()
            # ignore landmark
            boxes, boxes_c, landmark = self.detect_pnet(im)
           # t1 = time.time() - t
           # sum_time += t1
            if boxes_c is None:
                print("boxes_c is None...")
                all_boxes.append(np.array([]))
                # pay attention
                landmarks.append(np.array([]))


        # rnet

        if boxes_c is None:
            print('boxes_c is None after Pnet')
        t2 = 0
        if self.rnet_detector and not boxes_c is  None:
           # t = time.time()
            # ignore landmark
            boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
           # t2 = time.time() - t
           # sum_time += t2
            if boxes_c is None:
                all_boxes.append(np.array([]))
                landmarks.append(np.array([]))


        # onet
        t3 = 0
        if boxes_c is None:
            print('boxes_c is None after Rnet')

        if self.onet_detector and not boxes_c is  None:
          #  t = time.time()
            boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
         #   t3 = time.time() - t
          #  sum_time += t3
            if boxes_c is None:
                all_boxes.append(np.array([]))
                landmarks.append(np.array([]))


        #print(
         #   "time cost " + '{:.3f}'.format(sum_time) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3))

        all_boxes.append(boxes_c)
        landmarks.append(landmark)

        return all_boxes, landmarks