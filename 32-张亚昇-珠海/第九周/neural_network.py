import numpy as np
import scipy.special as T

class NeuralNetwork():
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.innodes = inputnodes
        self.hdnodes = hiddennodes
        self.outnodes = outputnodes
        self.lr = learningrate
        #初始化权重矩阵
        self.wih = np.random.rand(self.hdnodes, self.innodes) - 0.5
        self.who = np.random.rand(self.outnodes, self.hdnodes) - 0.5
        #激活函数
        self.activation_function = lambda x:T.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        #根据输入的训练数据更新节点链路权重
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        #计算信号经过输入层后产生的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))
        self.who += self.lr * np.dot((output_errors * final_outputs * (1-final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),
                                     np.transpose(inputs))

        pass

    def query(self, inputs):
        # 根据输入数据计算并输出答案
        hidden_inputs = np.dot(self.wih, inputs)
        #计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算最外层接收到的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs

#初始化网络
#输入图像为28*28=784
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#读入训练数据
training_data_file = open("./dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#加入epoch，设定训练循环次数
epochs = 50
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 +0.01
        #设置图像于数值的对应关系
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

test_data_file = open("./dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []

for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("g该图片对应的数字为：", correct_number)
    #预处理数字图像
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print("网络认为图像的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算准确率
scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)




