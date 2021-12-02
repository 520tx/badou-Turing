'''
第九周作业：
手动实现神经网络
'''
import numpy
import scipy.special
class  NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        #权重初始化
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5
        #self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        #self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))
        #激活函数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def  query(self,inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def train(self,inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets- final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors )
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),numpy.transpose(inputs))
        pass


training_data_file = open("dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

epochs = 5
for e in range(epochs):
    #把数据依靠','区分，并分别读入
    print ("第",e+1,"次训练开始")
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
    test_data_file = open("dataset/mnist_test.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_number = int(all_values[0])
        print("该图片对应的数字为:", correct_number)
        # 预处理数字图片
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 让网络判断图片对应的数字
        outputs = n.query(inputs)
        # 找到数值最大的神经元对应的编号
        label = numpy.argmax(outputs)
        print("网络认为图片的数字是：", label)
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)
    print(scores)
    # 计算图片判断的成功率
    scores_array = numpy.asarray(scores)
    print("perfermance = ", scores_array.sum() / scores_array.size)
    print("第", e+1, "次训练结束")
    e=e+1
# test_data_file = open("dataset/mnist_test.csv")
# test_data_list = test_data_file.readlines()
# test_data_file.close()
# scores = []
# for record in test_data_list:
#     all_values = record.split(',')
#     correct_number = int(all_values[0])
#     print("该图片对应的数字为:",correct_number)
#     #预处理数字图片
#     inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
#     #让网络判断图片对应的数字
#     outputs = n.query(inputs)
#     #找到数值最大的神经元对应的编号
#     label = numpy.argmax(outputs)
#     print("网络认为图片的数字是：", label)
#     if label == correct_number:
#         scores.append(1)
#     else:
#         scores.append(0)
# print(scores)
#
# #计算图片判断的成功率
# scores_array = numpy.asarray(scores)
# print("perfermance = ", scores_array.sum() / scores_array.size)
