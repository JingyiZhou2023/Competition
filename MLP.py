# -*- coding: utf-8 -*-
# Time : 2023/6/25 15:56
# Author : bently
# File : MLP.py
# Desc :
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from DataPreparation import train_x, train_y, test_data
from sklearn.preprocessing import StandardScaler


# 数据转换
def cuisine_to_y(cuisine):
    country_name = ['chinese', 'korean', 'indian', 'thai', 'japanese']
    return country_name.index(cuisine)


def y_to_cuisine(y):
    country_name = ['chinese', 'korean', 'indian', 'thai', 'japanese']
    return country_name[y]


# 准确率计算
def accuracy_metric(actual, predict):
    # 记录预测正确个数
    correct = 0
    for i in range(len(actual)):
        if actual.tolist()[i] == predict[i]:
            correct += 1
    return float(correct) / len(actual.tolist())


# 分割训练集验证集
x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.3, random_state=42)
# 数据处理
input = torch.t(torch.FloatTensor(x_train.values.tolist()))
valid = torch.t(torch.FloatTensor(x_valid.values.tolist()))
test = torch.t(torch.FloatTensor(test_data.values.tolist()))
output = torch.rand(0, 0)
for i in range(len(x_train)):
    yi = cuisine_to_y(y_train.iloc[i])
    output_i = [0, 0, 0, 0, 0]
    output_i[cuisine_to_y(y_train.iloc[i])] = 1
    output_i = torch.FloatTensor(output_i).reshape(-1, 1)
    if i == 0:
        output = output_i
    else:
        output = torch.cat((output, output_i), 1)
# 正交化
scale = StandardScaler()
input = torch.from_numpy(scale.fit_transform(input)).float()
valid = torch.from_numpy(scale.fit_transform(valid)).float()
test = torch.from_numpy(scale.fit_transform(test)).float()
# output = torch.from_numpy(scale.fit_transform(output)).float()


# 构造MLP(三层)
class MLPClassifier:
    # 隐藏层神经元个数
    hidden_cell_sizes = 30
    # 隐藏层层数
    hidden_layer_sizes = 1
    # 激活函数
    activation = torch.sigmoid
    input = torch.rand(0, 0)
    output = torch.rand(0, 0)
    results = torch.rand(0, 0)
    # 隐藏层输出
    a = torch.rand(0, 0)
    # 误差
    mse = []
    mse_final = 0.000001
    # 权重
    W = []
    # 偏移量
    b = []
    # 学习率
    lr = 0.01
    # 终止次数
    counts = 200000

    # 初始化
    def __init__(self, activation=torch.sigmoid, hidden_layer_sizes=1):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        for i in range(hidden_layer_sizes+1):
            self.W.append(None)
        for i in range(hidden_layer_sizes+1):
            self.b.append(None)

    # 反向传播
    def back_propagate(self):
        m = self.results.shape[1]
        dz_2 = torch.t(self.output - self.results)  # (m, n3)
        dw2 = 1 / m * torch.mm(self.a, dz_2)  # (n2, n3)
        db2 = 1 / m * torch.sum(dz_2, dim=0)
        # 梯度下降
        self.W[1] = self.W[1] + self.lr * torch.t(dw2)
        self.b[1] = self.b[1] + self.lr * db2.reshape(-1, 1)
        dz_1 = torch.mm(dz_2, self.W[1]) * ((1 - torch.t(self.a)) * torch.t(self.a))  # sigmoid(z_1)的导数  # (m, n2)
        dw1 = 1 / m * torch.mm(self.input, dz_1)  # (n1, n2)
        db1 = 1 / m * torch.sum(dz_1, dim=0)  # (1, n2)
        self.W[0] = self.W[0] + self.lr * torch.t(dw1)
        self.b[0] = self.b[0] + self.lr * db1.reshape(-1, 1)

    # 模型训练
    def fit(self, x, y):
        self.input = x
        self.output = y
        for c in range(self.counts):
            n_in = self.input
            # 初始化权重、偏移量
            for i in range(self.hidden_layer_sizes):
                Wi = self.W[i]
                bi = self.b[i]
                if Wi is None:
                    Wi = torch.zeros(self.hidden_cell_sizes, n_in.size()[0], dtype=torch.float)
                    self.W[i] = Wi
                if bi is None:
                    bi = torch.zeros(self.hidden_cell_sizes, 1, dtype=torch.float)
                    self.b[i] = bi
                self.a = self.activation(torch.mm(Wi, n_in) + bi)
                n_in = self.a
            Wh = self.W[len(self.W) - 1]
            bh = self.b[len(self.b) - 1]
            if Wh is None:
                Wh = torch.zeros(self.output.size()[0], n_in.size()[0], dtype=torch.float)
                self.W[len(self.W) - 1] = Wh
            if bh is None:
                bh = torch.zeros(self.output.size()[0], 1, dtype=torch.float)
                self.b[len(self.b) - 1] = bh
            self.results = torch.sigmoid(torch.mm(Wh, n_in) + bh)
            # 误差
            self.mse.append(np.average(np.square(self.results - self.output)))
            # 优化参数
            if self.mse[c] < self.mse_final:
                break
            else:
                self.back_propagate()

    # 预测
    def predict(self, test):
        n_in = test
        for i in range(self.hidden_layer_sizes):
            Wi = self.W[i]
            bi = self.b[i]
            ai = self.activation(torch.mm(Wi, n_in) + bi)
            n_in = ai
        Wh = self.W[len(self.W) - 1]
        bh = self.b[len(self.b) - 1]
        self.results = torch.sigmoid(torch.mm(Wh, n_in) + bh)
        max_rate, idx = self.results.max(0)
        predictions = []
        for i in range(len(idx)):
            predictions.append(y_to_cuisine(idx[i]))
        return predictions


# 模型训练
mlp = MLPClassifier()
mlp.fit(input, output)
# 准确率计算
max_rate, idx = mlp.results.max(0)
y = []
for i in range(len(idx)):
    y.append(y_to_cuisine(idx[i]))
print(f'accuracy of MLP classifier on training set is :{accuracy_metric(y_train, y)}')
print(f'accuracy of MLP classifier on test set is :{accuracy_metric(y_valid, mlp.predict(valid))}')

# 预测
mlp_predictions = pd.DataFrame(data=[], columns=['id', 'cuisine'])
predictions = mlp.predict(test)
for i in range(len(predictions)):
    mlp_predictions.loc[i] = [i+1, predictions[i]]
mlp_predictions.to_csv(r"mlp_submission.csv", index=False)

