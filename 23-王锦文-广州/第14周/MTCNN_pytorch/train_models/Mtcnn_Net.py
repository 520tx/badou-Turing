import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)
 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias, 0.1)

#损失函数
class LossFn:
    def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1):
        # loss function
        self.cls_factor = cls_factor#分类比重
        self.box_factor = box_factor#box回归比重
        self.land_factor = landmark_factor#land_mark比重
        #self.loss_cls = nn.BCELoss() # binary cross entropy  使用nn.BCELoss需要在该层前面加上Sigmoid函数。
        self.loss_cls=nn.CrossEntropyLoss(reduction='none')
        self.loss_box = nn.MSELoss() # mean square error
        self.loss_landmark = nn.MSELoss()

    # cls loss 只考虑 label 1和0
    # gt_label tensor([1,0,-1,-2])--batch_size
    # pred_label (batch_size,2,1,1)
    def cls_loss(self,gt_label,pred_label):
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
        
        # print(valid_gt_label.shape,valid_pred_label.shape)
        # valid_gt_label = torch.masked_select(gt_label,mask)#[1,0]
        # print(valid_gt_label)
        # print(valid_pred_label)
        # valid_pred_label = torch.masked_select(pred_label,mask)#(2,2)
        return self.loss_cls(valid_pred_label,valid_gt_label)

    # box loss 只考虑 label 1和-1 
    # gt_label tensor([1,0,-1,-2])--batch_size
    # gt_offset (batch_size,4)---获取数据时需要的shape
    # pred_offset (batch_size,4,1,1)
    def box_loss(self,gt_label,gt_offset,pred_offset):
        
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



        return self.loss_box(valid_pred_offset,valid_gt_offset)

    # cls loss 只考虑 label -2
    # gt_label tensor([1,0,-1,-2])--batch_size
    # gt_landmark (batch_size,10)---获取数据时需要的shape
    # pred_landmark (batch_size,10,1,1)
    def landmark_loss(self,gt_label,gt_landmark,pred_landmark):
        
        pred_landmark = torch.squeeze(pred_landmark)#去掉维数为1的维度(batch_size,10)
        gt_landmark = torch.squeeze(gt_landmark) #(batch_size,10)
        gt_label = torch.squeeze(gt_label)#(batch_size)--example [1,0,-1,-2]
        # print("333",gt_label)
        # print("333",gt_landmark)
        # print("333",pred_landmark)

        mask = torch.eq(gt_label,-2)#找到等于-2的索引 [0,0,0,1]
        chose_index = torch.nonzero(mask.data)#[[3]]
        chose_index = torch.squeeze(chose_index)#[3]

        valid_gt_landmark = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]
        # print("333",valid_gt_landmark)
        # print("333",valid_pred_landmark)
        return self.loss_landmark(valid_pred_landmark,valid_gt_landmark)

#P网络
class PNet(torch.nn.Module):
    def __init__(self,phase="train"):
        super(PNet, self).__init__()
        self.phase=phase
        # backend
        self.pre_layer = nn.Sequential(
                # pw
                nn.Conv2d(3, 10, kernel_size=3, stride=1, bias=False),
                # nn.BatchNorm2d(10),
                nn.PReLU(),
                # dw
                nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
                nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
                # nn.BatchNorm2d(16),
                nn.PReLU(),
                # pw-linear
                nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
                # nn.BatchNorm2d(32),
                nn.PReLU()
            )
        # detection
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        # self.softmax = nn.Softmax(dim=1)
        # assert self.conv4_1.shape[1]==2 and self.conv4_1.shape[2]==1 and self.conv4_1.shape[3]==1,"conv4_1 error"
        
        # bounding box regresion
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        # assert self.conv4_2.shape[1]==4 and self.conv4_2.shape[2]==1 and self.conv4_2.shape[3]==1,"conv4_2 error"
        
        # landmark localization
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)
        # weight initiation with xavier
        self._initialize_weights()
    
    def forward(self,x):
        x = self.pre_layer(x)
        # assert self.pre_layer.shape[1]==32 and self.pre_layer.shape[2]==1 and self.pre_layer.shape[3]==1,"pre_layer error"
        label = self.conv4_1(x)  #（batch_size,2,1,1）
        # label = self.softmax(label)  #（batch_size,2,1,1）
        if self.phase!="train":
            label = F.softmax(label,dim=1)
        #label = torch.sigmoid(label)
        box = self.conv4_2(x) #（batch_size,4,1,1）
        landmark = self.conv4_3(x)#（batch_size,10,1,1）
        return label, box,landmark
    # def _initialize_weights(self):
    #     for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #                 m.weight.data.normal_(0, math.sqrt(2. / n))
    #                 if m.bias is not None:
    #                     m.bias.data.zero_()
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 m.weight.data.fill_(1)
    #                 m.bias.data.zero_()
    #             elif isinstance(m, nn.Linear):
    #                 m.weight.data.normal_(0, 0.01)
    #                 m.bias.data.zero_()
    def _initialize_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform(m.weight.data)
        #         nn.init.constant(m.bias, 0.1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    # nn.init.constant(m.bias, 0.1)
                    nn.init.constant_(m.bias, 0.)
                    # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

#R网络
class RNet(torch.nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        # backend
        self.pre_layer = nn.Sequential(
                # pw
                nn.Conv2d(3,28, kernel_size=3, stride=1, bias=False),#conv1
                nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
                nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2
                nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
                nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
            )
        self.last_channel=2*2*64
        # detection
        self.liner1=nn.Linear(self.last_channel, 128)
        self.liner1_1=nn.Linear(128, 2)
        self.liner1_2=nn.Linear(128, 4)
        self.liner1_3=nn.Linear(128, 10)

        self._initialize_weights()
    
    def forward(self,x):
        x = self.pre_layer(x)
        #print(x.shape)torch.Size([5, 64, 2, 2])
        x=x.view(-1,self.last_channel)#(batch_size,3*3*64)
        x=self.liner1(x)#(batch_size,128)
        cls_pred=self.liner1_1(x)#(batch_size,2)
        box_pred=self.liner1_2(x)#(batch_size,4)
        landmark_pred=self.liner1_3(x)#(batch_size,10)
        return cls_pred, box_pred,landmark_pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()

#O网络
class ONet(torch.nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        # backend
        self.pre_layer = nn.Sequential(
                # pw
                nn.Conv2d(3,32, kernel_size=3, stride=1, bias=False),#conv1
                nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
                nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
                nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
                nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
                nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
                nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4
            )
        self.last_channel=2*2*128
        # detection
        self.liner1=nn.Linear(self.last_channel, 256)
        self.liner1_1=nn.Linear(256, 2)
        self.liner1_2=nn.Linear(256, 4)
        self.liner1_3=nn.Linear(256, 10)

        self._initialize_weights()
    
    def forward(self,x):
        x = self.pre_layer(x)
        x=x.view(-1,self.last_channel)#(batch_size,3*3*256)
        x=self.liner1(x)#(batch_size,256)
        cls_pred=self.liner1_1(x)#(batch_size,2)
        box_pred=self.liner1_2(x)#(batch_size,4)
        landmark_pred=self.liner1_3(x)#(batch_size,10)
        return cls_pred, box_pred,landmark_pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def main():
    rnet=RNet()
    inputs=torch.ones((5,3,24,24))
    rnet(inputs)
    
if __name__ == "__main__":
    main()
    