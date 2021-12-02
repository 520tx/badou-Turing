from model.AlexNet import AlexNet
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.optimizers import Adam
from dataset import generate_dataset_batch

# from model.Rsenet import Resnet50
# model = Resnet50(input_shape=(224, 224, 3), classes=2)

# from model.Vgg import Vgg16
# model = Vgg16(input_shape=(224, 224, 3), classes=2)

model = AlexNet(input_shape=(224, 224, 3), classes=2)

log_dir = r"./logs"

# 保存checkpoint的方式，每3个epoch保存一次
checkpoint_period = ModelCheckpoint(
            filepath=log_dir + '/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor="acc",   # val_acc 或 val_loss 或 acc 或 loss
            verbose=1,
            save_best_only=True,      # 验证集上acc最高的checkpoint不会被覆盖
            save_weights_only=False,   # model.save(filepath), model.save_weights(filepath)
            period=3)

# 学习率下降的方式，acc三次不下降就下降学习率继续训练
reduce_lr = ReduceLROnPlateau(
    monitor='acc',
    factor=0.5,   # new_lr = lr * factor
    patience=3,   # acc三次不下降就下降学习率
    verbose=1
)

# 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,  # val_loss下降值大于min_delta时，才算一次下降
    patience=10,        # val_loss连续10次不下降时，就停止训练
    verbose=1,
)

model.compile(
    optimizer=Adam(lr=0.0001),    # Adam优化器，学习率维0.001
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

# 一次的训练集大小
batch_size = 128
img_dir = r"D:\BaiduNetdiskDownload\【11】图像识别\代码\alexnet\train"
label_file = r"D:\BaiduNetdiskDownload\【11】图像识别\代码\alexnet\AlexNet-Keras-master\data\dataset.txt"

train_generator, train_num, val_generator, val_num = generate_dataset_batch(
    img_dir,
    label_file,
    batch_size,
    num_classes=2)

# model.fit()
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=max(1, train_num//batch_size),
    validation_data=val_generator,
    validation_steps=max(1, val_num/batch_size),
    epochs=50,
    initial_epoch=0,
    callbacks=[checkpoint_period, reduce_lr, early_stopping]
)

model.save_weights(log_dir+"last1.h5") # 进保存权值信息
# model.save() # 保留权值和网络信息
