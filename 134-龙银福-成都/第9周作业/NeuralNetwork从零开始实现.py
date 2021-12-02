import numpy as np
import scipy.special as sigmod
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        # 初始化权重，值域为[-0.5, 0.5]
        # self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        # self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数：sigmod函数
        self.activation_function = lambda x:sigmod.expit(x)

        # 均方差损失loss
        self.mse_loss = []

        pass

    def train(self, inputs_list, labels_list):
        # 改变维度
        inputs = np.array(inputs_list, ndmin=2).T
        labels = np.array(labels_list, ndmin=2).T

        # 前向推理
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差，反向传播法 --> 更新权值
        output_errors = labels - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        self.mse_loss = (np.power((labels - final_outputs), 2)).sum() * 0.5
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                     np.transpose(inputs))
        return self.mse_loss

        pass

    def query(self, inputs):
        # 输入层 --> 隐藏层
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 隐藏层 --> 输出层
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        print(final_outputs)
        return final_outputs

        pass

if __name__ == '__main__':
    [3]
    # 1-初始化网络模型
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3
    mse_loss = []
    epoch_loss = []
    model = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # 2-加载数据 --> 数据预处理
    train_data_file = open('dataset/mnist_train.csv')
    train_data_list = train_data_file.readlines()
    train_data_file.close()
    # 3-训练模型
    epochs = 20
    for e in range(epochs):
        for record in train_data_list:
            all_train_value = record.split(',')
            inputs = (np.asfarray(all_train_value[1:])) / 255.0 * 0.99 + 0.01
            targets = np.zeros(output_nodes) + 0.01 # label --> one-hot形式
            targets[int(all_train_value[0])] = 0.99
            loss = model.train(inputs, targets)
            mse_loss.append(loss)
        loss = np.array(mse_loss)
        epoch_loss.append(loss.mean())
        mse_loss = []
    plt.plot(epoch_loss)
    plt.show()
    # 4-测试训练完成的模型效果
    test_data_file = open('dataset/mnist_test.csv')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []
    for record in test_data_list:
        all_test_value = record.split(',')
        correct_number = int(all_test_value[0])
        print(correct_number)
        inputs = (np.asfarray(all_test_value[1:])) / 255.0 * 0.99 + 0.01
        outputs = model.query(inputs)
        label = np.argmax(outputs)
        print('query: ', label)
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)
    print('scores: ', scores)
    # 5-计算准确率accuracy
    scores_array = np.asarray(scores)
    accuracy = scores_array.sum() / scores_array.size
    print('accuracy: ', accuracy)


    '''
    [2]
    # 从文件中加载数据
    data_file = open('dataset/mnist_test.csv')
    data_list = data_file.readlines()
    data_file.close()
    print(len(data_list))
    print(data_list[1])
    # 画图 --> 可视化
    all_values = data_list[1].split(',') # 以逗号为分隔符去分隔数据列表 (把数据依靠','区分，并分别读入)
    image_array = np.asfarray(all_values[1:]).reshape((28, 28)) # 第一个值是label, 删掉后再reshape
    plt.imshow(image_array, cmap='Greys', interpolation=None)
    plt.show()
    # 数据预处理 --> 归一化，值域[0.01, 1]
    scaled_input = image_array / 255.0 * 0.99 +0.01
    print(scaled_input)
    # label数据预处理 --> label 转化成 one-hot 形式
    onodes = 10
    targets = np.zeros(onodes) + 0.01
    targets[int(all_values[0])] = 0.99
    print(targets)
    '''

    '''
    [1]
    # 测试 query() 函数是否存在问题
    inputnotes = 3
    hiddennotes = 3
    outputnotes = 3
    learningrate = 0.3
    n = NeuralNetwork(inputnotes, hiddennotes, outputnotes, learningrate)
    n.query([1.0, 0.5, -1.5])
    '''