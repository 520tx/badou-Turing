#coding=utf-8
import tensorflow as tf 
import numpy as np
'''
该文件使用tensorflow构建简单网络,运行环境是tf1.15
'''
def appendlayer(x,inchannels,outchannels,activation=tf.nn.sigmoid):
    '''
    添加网络层
    x:输入的训练数据
    inchannels：输入通道数
    outchannels：输出通道数
    activateion：激活函数类型
    '''
    #随机初始化权重
    W=tf.Variable(tf.truncated_normal([inchannels,outchannels],mean=0, stddev=0.1))
    bias=tf.Variable(tf.zeros([outchannels]))
    Wx_plus_b=tf.matmul(x,W)+bias
    if activation is not None:
        output=activation(Wx_plus_b)   #加入激活函数
    else:
        output=Wx_plus_b
    return output


def build_model(input,cfg=[784,200,10],activation=tf.nn.sigmoid):
    '''
    构建网络，返回输出
    '''
    output=input
    for i in range(len(cfg)-1):
        activation=tf.nn.softmax if (i==len(cfg)-1) else activation#
        output=appendlayer(output,cfg[i],cfg[i+1],activation=activation)
    return output

def load_data(path):
    '''
    读取数据集
    '''
    with open(path,'r') as f:
        training_data_list = f.readlines()
    all_inputs=[]
    all_targets=[]
    for line in training_data_list:
        values=line.split(",")
        inputs = (np.asfarray(values[1:]))/255.0 * 0.99 + 0.01
        targets=int(values[0])#标签
        all_inputs.append(inputs)
        all_targets.append(targets)
    all_inputs=np.array(all_inputs)
    all_targets=np.array(all_targets)
    all_targets=all_targets.reshape(all_inputs.shape[0],1)
    total_inputs=np.concatenate((all_inputs,all_targets),axis=1)
    return total_inputs

def train(datapath,epochs=8,batch_size=4):
    '''
    训练
    datapath：训练数据路径
    batch_size:batch大小
    '''
    
    #定义两个placeholder存放输入数据
    x=tf.placeholder(tf.float32,[None,784])
    y=tf.placeholder(tf.float32,[None,10])
    #预测输出
    output=build_model(x)

    #定义loss，
   # loss=-tf.reduce_sum(y*tf.log(output),reduction_indices=[-1])
    loss =tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(y, output))))
    #定义反向传播算法（使用梯度下降算法训练）
    #train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    train_step=tf.train.AdamOptimizer(0.001).minimize(loss)
    #必须初始化所有变量
    init= tf.initialize_all_variables()

    #对数据进行读取
    total_inputs=load_data(datapath)
    #获取一个epoch总的batch数量
    max_batch_len=int(total_inputs.shape[0]/batch_size)
    with tf.Session() as sess:
        sess.run(init)
        #训练
        for  i in range(epochs):
            np.random.shuffle(total_inputs)#打乱数据
            for bs in range(max_batch_len):
                #获取一个batch的数据
                images=total_inputs[bs*batch_size:(bs+1)*batch_size,:-1]
                labels=total_inputs[bs*batch_size:(bs+1)*batch_size,-1].astype(np.int32)
                #将标签变为one-hot
                targets_onehot=np.zeros([batch_size,10])+0.01
                targets_onehot[np.arange(batch_size),labels]=0.99
                result=sess.run([train_step,loss],feed_dict={x:images,y:targets_onehot})
                print("epoch:{},loss={}".format(i,result[1]))
                
        #训练完成后对数据进行测试工作,
        test_data_list=load_data("../dataset/mnist_test.csv")
        all_predlabel=[]
        for i in range(test_data_list.shape[0]):
            img,label=test_data_list[i][:-1],test_data_list[i][-1]
            img=np.expand_dims(img,axis=0)
            #预测
            out=sess.run(output,feed_dict={x:img})
            out=np.argmax(out)
            print("图片真实值是：{}，预测值是：{}".format(label,out))
            all_predlabel.append(out)
        gt=test_data_list[:,-1]#真实标签
        acc=sum((all_predlabel==gt))/len(gt)
        print("testset accuracy :",acc)

if __name__=='__main__':
    train("../dataset/mnist_train.csv")




    
        







        

