import argparse
import os
import sys
sys.path.insert(0,"..")
from Mtcnn_Net import PNet,RNet 
import torch
import torch.nn as nn
from Detection.Mtcnn_Detector import MtcnnDetector
# from models import PNet
# from Nets import PNet




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='生成R网络和O网络的数据',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', help='',default='/media/ubuntu_data2/02_dataset/wjw/dataset/tf_MTCNN_DATA/', type=str)
parser.add_argument('--load_pnet_path', help='test net type, can be pnet, rnet or onet',default='../train_models/Pnet_weight', type=str)
parser.add_argument('--load_rnet_path', help='test net type, can be pnet, rnet or onet',default='../train_models/Rnet_weight', type=str)
parser.add_argument('--load_onet_path', help='test net type, can be pnet, rnet or onet',default='../train_models/Onet_weight', type=str)

#parser.add_argument('--prefix', dest='prefix', help='prefix of model name', default=['../train_models/Pnet_weight', '../train_models/Rnet_weight', '../train_models/Onet_weight'],type=str)
#parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load',default=[19, 19, 19], type=int)
#parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction',default=[2048, 256, 16], type=int)
parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet',default=[0.9, 0.6, 0.7], type=float)
parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',default=30, type=int)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--use_gpu', type=int, default=1,help='use gpu')
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
    

def get_img_list(data_dir):
    list_img=[]
    for root,dirs,files in os.walk(args.data_dir,topdown=False):
        for file_name in files:
            if file_name.endswith(".jpg"):
                img_full_path=os.path.join(root, file_name)
                list_img.append(img_full_path)
    return list_img

def test_net(thresh=[0.6, 0.6, 0.7], min_face_size=25):
    print("the thresh: ",thresh)
    #获得模型路径
    print("the pnet model_path ",args.load_pnet_path)

    print("**********start load pnet***********")
    ret,p_model=load_model("PNet",args.load_pnet_path)
    if not ret:
        print("**********fail load pnet***********")
        return
    print("**********load pnet successfully***********")

 
    print("**********start load Rnet***********")
    ret,r_model=load_model("RNet",args.load_rnet_path)
    if not ret:
        print("**********fail load RNet***********")
        return
    print("**********load RNet successfully***********")

    # print("**********start load Onet***********")
    ret,o_model=load_model("ONet",args.load_onet_path)
    if not ret:
        print("**********fail load onet***********")
        return
    print("**********load onet successfully***********")
    
    img_list=get_img_list(args.data_dir)

    #创建检测器对象
    mtcnn_detector = MtcnnDetector(pnet=p_model,rnet=r_model,onet=o_model, min_face_size=min_face_size, threshold=thresh,use_gpu=args.use_gpu)
    # detections,_ = mtcnn_detector.detect_face(dataloader,datasize)
    
    print ('finish detecting ')
    for imgname in img_list:
        img=cv2.imread(imgname)
        boxes_c,landmarks = mtcnn_detector.detect(img)
        print(landmarks.shape)
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # if score > thresh:
            cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
            cv2.putText(img, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
        cv2.putText(img, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2)
        for i in range(landmarks.shape[0]):
            for j in range(int(len(landmarks[i])/2)):
                cv2.circle(img, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))            
        # time end
        cv2.imshow("", img)
        # print(cv2.waitKey(1) & 0xFF)
        # print(ord('q'))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


#python  test.py  --data_dir   --load_pnet_path ../train_models/PNet_debug/checkpoints_epoch_28.tar --load_rnet_path ../train_models/RNet_weight/iter_41249.pth.tar --load_onet_path ../train_models/ONet_weight/iter_41249.pth.tar
def main():
    test_net(thresh=args.thresh, min_face_size=args.min_face)

if __name__=='__main__':
    main()