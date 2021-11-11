import numpy as np
import numpy.random as npr
import os
import argparse

parser = argparse.ArgumentParser(description="prepare landmark data for mtcnn pnet")
parser.add_argument('--net_name', type=str, default="PNet",help="生成landmark的网络名称")
parser.add_argument('--data_dir', type=str, default='/media/ubuntu_data2/02_dataset/wjw/dataset/torch_MTCNN_DATA/12',help="生成landmark的网络名称")

args = parser.parse_args()

def main():
    if args.net_name == "PNet":
        size = 12
    if args.net_name == "RNet":
        size = 24
    if args.net_name == "ONet":
        size = 48

    with open(os.path.join(args.data_dir, 'pos_{}.txt'.format(size)), 'r') as f_pos:
        pos = f_pos.readlines()

    with open(os.path.join(args.data_dir, 'neg_{}.txt'.format(size)), 'r') as f_neg:
        neg = f_neg.readlines()

    with open(os.path.join(args.data_dir, 'part_{}.txt'.format(size)), 'r') as f_part:
        part = f_part.readlines()

    with open(os.path.join(args.data_dir,'landmark_{}_aug.txt'.format(size)), 'r') as f_land:
        landmark = f_land.readlines()
        
    # dir_path = os.path.join(args.data_dir, 'imglists')
    # dst_dir=os.path.join(dir_path, "%s" %(args.net_name))
    # if not os.path.exists(dst_dir):
    #     os.makedirs(dst_dir)
    with open(os.path.join(args.data_dir,"train_%s_imglist.txt" % (args.net_name)), "w") as f_list:
        nums = [len(neg), len(pos), len(part)]
        ratio = [3, 1, 1]
        #base_num = min(nums)
        base_num = 1300000
        print("打乱数据前 neg样本个数:{} pos样本个数:{} part样本个数:{} 基础样本个数:{}".format(len(neg), len(pos), len(part), base_num))

        #shuffle the order of the initial data
        #if negative examples are more than 750k then only choose 750k
        if len(neg) > base_num * 3:
            neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)
        else:
            neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
        pos_keep = npr.choice(len(pos), size=base_num, replace=True)
        part_keep = npr.choice(len(part), size=base_num, replace=True)
        print("打乱数据后 neg样本个数:{} pos样本个数:{} part样本个数:{}".format(len(neg_keep), len(pos_keep), len(part_keep)))

        # write the data according to the shuffled order
        for i in pos_keep:
            f_list.write(pos[i])
        for i in neg_keep:
            f_list.write(neg[i])
        for i in part_keep:
            f_list.write(part[i])
        for item in landmark:
            f_list.write(item)
        # 打乱数据前 neg样本个数:1000079 pos样本个数:458585 part样本个数:1127972 基础样本个数:250000
        # 打乱数据后 neg样本个数:750000 pos样本个数:250000 part样本个数:250000

if __name__ == "__main__":
    main()


