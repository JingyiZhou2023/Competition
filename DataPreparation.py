# -*- coding: utf-8 -*-
# Time : 2023/6/28 10:11
# Author : bently
# File : DataPreparation.py
# Desc :
import pandas as pd
from sklearn.model_selection import train_test_split

# 对标签的分布以饼状图形式进行可视化展示
# 将训练集中数据读取出来
train_df = pd.read_csv(r'recipes_train.csv')
# 拆分训练数据
# 训练数据
train_x = train_df.drop('cuisine', axis=1).drop('id', axis=1)
train_y = train_df["cuisine"]
# 分割训练集验证集
x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
# 测试数据
test_data = pd.read_csv(r'recipes_test.csv').drop('id', axis=1)


# 准确率计算
def accuracy_metric(actual, predict):
    # 记录预测正确个数
    correct = 0
    for i in range(len(actual)):
        if actual.tolist()[i] == predict.tolist()[i]:
            correct += 1
    return float(correct) / len(actual.tolist())
