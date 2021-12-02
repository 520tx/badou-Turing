from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import AlexNet
import numpy as np
import utils
import cv2

# K.set_image_dim_ordering('tf') # Keras 1.x中的 image_dim_ordering, 已弃用
# 返回默认图像的维度顺序
# 'channels_last' 对应原本的'tf'  --> [Width, Height, Channels]
# 'channels_first'对应原本的'th'  --> [Channels, Width, Height]
K.image_data_format() == 'channels_first'

# 生成器函数，生成器的输出应该为(inputs, targets)的tuple元组
def generate_arrays_from_file(lines, batch_size):
    n = len(lines)      # 获取总长度
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r".\data\image\train" + '/' + name)    # 原始字符串：在字符串的第一个引号前加上字母 r（可以大小写），所有的字符串都是直接按照字面的意思来使用，没有转义特殊或不能打印的字符
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)              # BGR 转化为 RGB
            img = img / 255         # 归一化
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i + 1) % n
        # 处理图像
        X_train = utils.resize_image(X_train, (224, 224))
        X_train = X_train.reshape(-1, 224, 224, 3)
        # Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
        Y_train = to_categorical(np.array(Y_train), num_classes=2)
        yield (X_train, Y_train)

if __name__ == "__main__":

    # [1] 加载训练数据
    log_dir = "./logs/" # 训练好的模型参数保存的位置
    with open(r".\data\dataset.txt", "r") as f: # 打开数据集的txt
        lines = f.readlines()

    np.random.seed(10101)       # 设置随机数种子，使得每次生成的随机数相同
    np.random.shuffle(lines)    # 打乱行，这个txt主要用于帮助读取数据来训练，打乱的数据更有利于训练
    np.random.seed(None)

    num_val = int(len(lines) * 0.1)     # 90%用于训练，10%用于估计
    num_train = len(lines) - num_val

    # [2] AlexNet模型初始化
    model = AlexNet()

    # [3] 设置训练过程的优化方法
    # 作用：该回调函数将在每个epoch后保存模型到filepath(第一个参数)
    # 保存的方式，3世代(epoch)保存一次
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',     # 字符串，保存模型的路径
        monitor='acc',              # 需要监视的值，通常为：val_acc 或 val_loss 或 acc 或 loss
        save_weights_only=False,    # 若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
        save_best_only=True,        # 当设置为True时，将只保存在验证集上性能最好的模型
        period=3                    # CheckPoint之间的间隔的epoch数
    )
    # 在训练过程中缩小学习率，使用keras中的回调函数ReduceLROnPlateau()
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',      # 监测的值，可以是accuracy，val_loss,val_accuracy
        factor=0.5,         # 缩放学习率的值，学习率将以lr = lr*factor的形式被减少
        patience=3,         # 当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
        verbose=1           # 信息展示模式，0 或 1：为1表示输出epoch模型保存信息，默认为0表示不输出该信息
    )
    # 目的：防止过拟合
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',     # 需要监视的量，val_loss，val_acc
        min_delta=0,            # 认为监控指标有提升的最小提升值；如果变化值小于该值，则认为监控指标没有提升
        patience=10,            # 当early stop被激活(如发现loss相比上一个epoch训练没有下降)，则经过patience个epoch后停止训练
        verbose=1               # 信息展示模式：为1表示输出epoch模型保存信息，默认为0表示不输出该信息
    )

    # [4] 模型编译
    model.compile(loss='categorical_crossentropy',  # 目标函数：交叉熵
                  optimizer=Adam(lr=1e-3),          # 优化器实例
                  metrics=['accuracy'])             # 在训练和测试期间的模型评估标准，通常使用metrics = ['accuracy']

    batch_size = 128 # 设置一次的训练集大小
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # [5] 开始训练模型
    # 利用Python的生成器，逐个生成数据的batch并进行训练，生成器与模型将并行执行以提高效率
    # 适合于训练数据集非常大的情况，不能同时将数据载入内存中
    model.fit_generator(
        generate_arrays_from_file(lines[:num_train], batch_size),   # generator：生成器函数，生成器的输出应该为(inputs, targets)的tuple元组
        steps_per_epoch=max(1, num_train // batch_size),            # int，当生成器返回steps_per_epoch次数据时一个epoch结束，执行下一个epoch
                                                                    # int(number_of_train_samples / batch_size)
        validation_data=generate_arrays_from_file(lines[num_train:], batch_size), # 验证集，与generator类似
        validation_steps=max(1, num_val // batch_size),
        epochs=50,          # int，训练的epoch数
        initial_epoch=0,    # 从该参数指定的epoch开始训练，在继续之前的训练时有用
        callbacks=[checkpoint_period1, reduce_lr]   # 设置业务需要的回调函数(其实是类)，所有的回调函数都要继承callback类或其子类
    )

    # [6] 保存训练好的模型权重参数
    model.save_weights(log_dir + 'last1.h5')