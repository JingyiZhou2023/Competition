# -*- coding: utf-8 -*-
# Time : 2023/6/28 10:07
# Author : bently
# File : DTModel.py
# Desc :
from DataPreparation import train_df, x_train, x_valid, y_valid, y_train, test_data, accuracy_metric
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from math import log


# 决策树
# 计算信息熵
def calc_shannon_ent(dataset):
    num_entries = len(dataset)
    label_counts = {}
    for info in dataset:
        current_label = info[1]
        if current_label not in label_counts:
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key])/num_entries
        shannon_ent -= prob*log(prob, 2)
    return shannon_ent


# 划分数据集
def split_data_set(dataset, idx, value):
    ret_data_set = []
    values = dataset.keys().tolist()
    for i in range(len(train_df)):
        if dataset.iloc[i][values[idx]] == value:
            reduced_info = dataset.iloc[i][:values[idx]].values.tolist()
            reduced_info.extend(dataset.iloc[i][values[idx+1]:].values.tolist())
            ret_data_set.append(reduced_info)
    return ret_data_set


info_D = calc_shannon_ent(train_df)
# 信息增益列表
gain_list = pd.DataFrame(data=[], columns=['food_name', 'gain'])
# 列名
food_list = train_df.keys().tolist()[2:]
# 统计列中1的个数
food_num = train_df.iloc[:, 0:].sum(axis=0).tolist()[2:]
# 计数
n = 0
for i in range(len(food_list)):
    # 将第i个特征按值为1划分数据集
    D1 = split_data_set(train_df, i, 1)
    # 将第i个特征按值为0划分数据集
    D0 = split_data_set(train_df, i, 0)
    # 计算信息熵
    info_D1 = calc_shannon_ent(D1)
    info_D0 = calc_shannon_ent(D0)
    # 计算第i个属性的信息熵
    info = float(food_num[i]) / len(train_df) * info_D1 + float(1 - food_num[i]) / len(train_df) * info_D0
    gain_list.loc[n] = [food_list[i], info_D - info]
    print(gain_list)
    n = len(gain_list.index)


# 决策树
# 模型训练
dt_clf = DecisionTreeClassifier().fit(x_train, y_train)
# 训练集中决策树分类器的精确度
print('Accuracy of Decision Tree classifier on training set: {:}'.format(accuracy_metric(dt_clf.predict(x_train), y_train)))
# 测试集中决策树分类器的精确度
print('Accuracy of Decision Tree classifier on test set: {:}'.format(accuracy_metric(dt_clf.predict(x_valid), y_valid)))
# 预测及保存
predictions = pd.DataFrame(data=[], columns=['id', 'cuisine'])
# 将结果保存至recipes_sample_submission.csv文件中
idx = 0
for info in dt_clf.predict(test_data).tolist():
    predictions.loc[idx] = [idx+1, info]
    idx = len(predictions.index)
predictions.to_csv(r"dt_submission.csv", index=False)
