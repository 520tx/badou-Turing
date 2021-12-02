#coding=utf-8
import tensorflow as tf
from model import *
from read_dataset import *
import argparse
import os
os.environ['CUDA_VISABLE_DEVICES']='0'
def train(args):
    #获取数据集
    traindata,valdata=get_dataset(args.datapath)
    #建立模型
    model=build_AlexNet()
    #学习率策略
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='acc', 
                            factor=0.6, 
                            patience=3, 
                            verbose=1
                        )
    
    #定义优化器
    sgd=tf.keras.optimizers.SGD(lr=args.lr,decay=5e-4,momentum=0.937,nesterov=True)
    #编译
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    model.summary()#输出量化模型的各层参数
    #训练
    model.fit_generator(generator(args.rootpath,traindata,args.batch_size),
            steps_per_epoch=max(1, len(traindata)//args.batch_size),
            validation_data=generator(args.rootpath,valdata,args.batch_size),
            validation_steps=max(1, len(valdata)//args.batch_size),
            epochs=args.epochs,
            initial_epoch=0,
            callbacks=[reduce_lr])
    model.save_weights(args.savepath+'model_last.h5')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='keras Alexnet train')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=50, type=int,
                             help='how many epochs to train')
    parser.add_argument('--batch_size', default=16, type=int,
                             help='batch_size')
    parser.add_argument('--numclasses', default=2, type=int,
                             help='how many classes')
    parser.add_argument('--rootpath', default='./train/', type=str,
                             help='where is the train dataset ')
    parser.add_argument('--datapath', default='./dataset.txt', type=str,
                             help='where is the train dataset change it to txt,like cat_0.jpg 0 ;dog_1.jpg 1')
    parser.add_argument('--savepath', type=str, default='./',
                        help='savemodel path')
    
    parser.add_argument('--changefileintotxt', action='store_true',
                        help='write train sample name and give its label and save to txt')

    args = parser.parse_args()
    #先生成txt,这里是先提前生成了
    if args.changefileintotxt:
        Writedataintotxt(args.rootpath)
    train(args)



