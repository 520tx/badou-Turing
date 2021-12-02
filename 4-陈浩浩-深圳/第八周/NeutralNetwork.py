# coding:utf-8
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_backward(dz, x):
    z = sigmoid(x)
    return dz * z * (1 - z)


def relu(x):
    x = np.copy(x)
    x[x <= 0] = 0  # relu: max(0, x)
    return x


def relu_backward(dz, x):
    dx = np.copy(dz)
    dx[x <= 0] = 0
    return dx


def cross_entropy_loss(pred, target):
    # target: (batch_size,), pred: (batch_size, nClass)
    label = np.zeros((target.shape[0], pred.shape[1])) # one-hot encoding，编码
    for i in range(target.shape[0]):
        # label[i, target[i]] = 1
        label[i, target[i]] = 0.99

    pred_sft = np.exp(pred)/(np.sum(np.exp(pred), axis=1)[:, None])  # softmax求概率
    loss = -np.sum(np.log(pred_sft)*label)             # crossEntropy 求交叉熵损失
    grad = cross_entropy_loss_backward(pred_sft, label)     # 求交叉熵梯度，反向传播使用
    return loss/pred.shape[0], grad    # loss/pred.shape[0]:是为了将整个batch的loss平均后返回，方便外层调用使用，
    # 注意：求导只是求-np.sum(np.log(pred_sft)*label)这一项的梯度, 这里不需要考虑batch_zie,后面backward过程中考虑了


def cross_entropy_loss_backward(pred_softmax, one_hot_label):
    return pred_softmax - one_hot_label
    # 详细推导过程：https://zhuanlan.zhihu.com/p/131647655


class Network(object):

    def __init__(self, net_architecture, learning_rate):
        assert len(net_architecture) > 0 and isinstance(net_architecture[0], dict), \
            print("wrong format of net_architecture:{}".format(net_architecture))
        self.params = {}  # 权值参数
        self.grads = {}  # 梯度
        self.cache = {}  # 缓存，方便backward propagation
        self.net_arch = net_architecture
        self.lr = learning_rate
        for idx, layer in enumerate(net_architecture):
            self.params["w{}".format(idx + 1)] = np.random.normal(0, pow(layer["output_dim"], -0.5),
                                                                  (
                                                                  layer["output_dim"], layer["input_dim"]))  # 初始化weight
            self.params["b{}".format(idx + 1)] = np.random.randn(layer["output_dim"], 1) * 0.1  # 初始化bias

    def train(self, batch_data, batch_target, loss_func="cross_entropy_loss"):
        pred = self.forward(batch_data)  # pred: shape(batch_size, nClass)
        if loss_func == "cross_entropy_loss":
            loss, loss_grad = cross_entropy_loss(pred, batch_target)  # loss为一个batch的平均loss
            self.backward(loss_grad)
        else:
            raise Exception("Unimplemented loss func")
        self.update()
        return loss

    def query(self, data):
        pred = self.forward(data)
        return np.argmax(pred, axis=1)   # shape(batch_size, )

    def forward_once(self, input_prev, w_cur, b_cur, activation="relu"):
        output_cur = np.dot(w_cur, input_prev) + b_cur
        if activation == "relu":
            activation_func = relu
        elif activation == "sigmoid":
            activation_func = sigmoid
        else:
            raise Exception("Unimplemented activation func")
        return activation_func(output_cur), output_cur

    def forward(self, x):
        input = x.T    # x shape : from (batch_size, input_dim) to (input_dim, batch_size)
        for idx, layer in enumerate(self.net_arch):
            w = self.params["w{}".format(idx+1)]
            b = self.params["b{}".format(idx+1)]
            output, output_cur = self.forward_once(input, w, b, activation=layer["activation_func"])

            self.cache["input{}".format(idx+1)] = input
            self.cache["output{}".format(idx+1)] = output_cur   # 储存wx+b，未经过激活函数的值
            input = output
        return output.T   # output shape : from (output_dim, batch_size) to (batch_size, output_dim)

    def backward_once(self, dx, w_cur, b_cur, input_cur, output_cur, activation="relu"):
        n = input_cur.shape[1]  # batch_size
        if activation == "relu":
            activation_backward = relu_backward
        elif activation == "sigmoid":
            activation_backward = sigmoid_backward
        else:
            raise Exception("Unimplemented activation func")
        activation_grad = activation_backward(dx, output_cur)
        bp_grad = np.dot(w_cur.T, activation_grad)

        # 注意！！！： weight_grad: shape(5 10), 和w_cur的shape相同，但这个梯度是4组数据(batch_size=4)的梯度之和，除4表示求整个batch的平均梯度
        weight_grad = np.dot(activation_grad, input_cur.T)/n

        # 注意！！！： b_cur:shape(5, 1); activation_grad:shape(5, 4); 这里的4表示batch_size, 求和除4，相当于求整个batch的平均梯度
        bias_grad = np.sum(activation_grad, axis=1, keepdims=True)/n

        return bp_grad, weight_grad, bias_grad

    def backward(self, dy):
        bp_grad_input = dy.T  # dy shape: from (batch_size, output_dim) to (output_dim, batch_size)
        for idx, layer in reversed(list(enumerate(self.net_arch))):
            w = self.params["w{}".format(idx + 1)]
            b = self.params["b{}".format(idx + 1)]
            input = self.cache["input{}".format(idx+1)]
            output = self.cache["output{}".format(idx+1)]
            bp_grad_output, weight_grad, bias_grad = self.backward_once(bp_grad_input, w, b, input, output, activation=layer["activation_func"])
            self.grads["weight_grad{}".format(idx + 1)] = weight_grad
            self.grads["bias_grad{}".format(idx + 1)] = bias_grad
            bp_grad_input = bp_grad_output

    def update(self):  # 梯度下降，更新权重参数
        for idx, layer in enumerate(self.net_arch):
            self.params["w{}".format(idx + 1)] -= self.lr*self.grads["weight_grad{}".format(idx + 1)]
            self.params["b{}".format(idx + 1)] -= self.lr*self.grads["bias_grad{}".format(idx + 1)]

def load_batch_data(training_data_list, index, batch_size):
    inputs = []
    targets = []
    for record in training_data_list[index:index+batch_size]:
        all_values = record.strip().split(',')
        img = np.asfarray(all_values[1:])/255.0*0.99+0.01
        inputs.append(img)
        targets.append(int(all_values[0]))
    return np.array(inputs), np.array(targets)

if __name__ == "__main__":
    net_architecture = [
        {"input_dim": 784, "output_dim": 200, "activation_func": "relu"},
        {"input_dim": 200, "output_dim": 100, "activation_func": "relu"},
        {"input_dim": 100, "output_dim": 10, "activation_func": "sigmoid"},
    ]

    learning_rate = 0.1
    net = Network(net_architecture, learning_rate)

    # open函数里的路径根据数据存储的路径来设定
    training_data_file = open("dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epoch = 100
    batch_size = 4
    loss_list = []
    for i in range(epoch):
        epoch_loss = 0
        for j in range(0, len(training_data_list), batch_size):
            batch_data, batch_target = load_batch_data(training_data_list, j, batch_size)
            # print(batch_data.shape,batch_target.shape)
            loss = net.train(batch_data, batch_target, loss_func="cross_entropy_loss")  # loss为一个batch的平均loss
            epoch_loss += loss
        epoch_loss = epoch_loss*batch_size/len(training_data_list)  # 一个epoch的平均loss
        loss_list.append(epoch_loss)
        print("[Epoch {}/{}] training loss: {:.4f}".format(i+1, epoch, epoch_loss))

    test_data_file = open("dataset/mnist_test.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    test_data = []
    test_target = []
    for record in test_data_list:
        all_values = record.split(',')
        test_target.append(int(all_values[0]))
        img = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        test_data.append(img)
    test_data = np.array(test_data)
    test_target = np.array(test_target)
    test_pred = net.query(test_data)
    print(test_target, test_pred)
    precision = np.sum(test_pred == test_target)/test_target.shape[0]
    print("Test precision: {:.4f}%".format(precision*100))

