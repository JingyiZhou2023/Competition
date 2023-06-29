# -*- coding: utf-8 -*-
# Time : 2023/6/28 10:36
# Author : bently
# File : ModelEnsemble_bagging.py
# Desc :
from DataPreparation import train_df, x_train, x_valid, y_valid, y_train, test_data, accuracy_metric
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import random
import pandas as pd


# 构造bagging分类器
class BaggingClassifier:
    # 分类器
    estimators = []
    # 随机样本
    samples = []
    # 训练数据集
    train_df = pd.DataFrame(data=[])
    # 测试数据集
    test_df = []
    # 样本中数据条数，若m>1，则为个数；若0<m<1，则为倍数
    m = 1
    # 投票方式，技术有限只写了硬投票
    method = 'hard'
    # 最终产生的预测结果
    result_prediction = pd.DataFrame(data=[], columns=['id', 'cuisine'])

    # 有放回的随机抽取m个样例形成子集
    # 若m > 1，则为m个；若0 < m < 1，则为len(dataset) * m个
    def random_sample(self, dataset, m):
        data = pd.DataFrame(data=[], columns=dataset.keys().tolist())
        if m < 1:
            for i in range(int(len(dataset) * m)):
                a = random.randint(0, len(dataset) - 1)
                data.loc[i] = dataset.iloc[a].tolist()
        else:
            for i in range(m):
                a = random.randint(0, len(dataset) - 1)
                data.loc[i] = dataset.iloc[a].tolist()
        return data

    # 生成n个随机样本
    def random_samples_n(self, dataset, m, n):
        samples = []
        for i in range(n):
            samples.append(self.random_sample(dataset, m))
        return samples

    # 硬投票
    def hard_voting(self, prediction_list):
        # 最终产生的预测结果
        result_prediction = pd.DataFrame(data=[], columns=['id', 'cuisine'])
        for i in range(len(prediction_list[0])):
            # print(i)
            # 属性值
            values_i = []
            # 数量
            counts_i = []
            # 对不同属性值投票计数
            for j in range(len(self.estimators)):
                # print(prediction_list[j][i])
                if prediction_list[j][i] not in values_i:
                    values_i.append(prediction_list[j][i])
                    counts_i.append(0)
                index = values_i.index(prediction_list[j][i])
                # print(index)
                counts_i[index] += 1
            # 选出票数最高的作为最终结果
            max_i = 0
            for k in range(len(counts_i)):
                if counts_i[max_i] <= counts_i[k]:
                    max_i = k
            # print(values_i, counts_i, max_i)
            result_prediction.loc[i] = [i+1, values_i[max_i]]
        return result_prediction

    # 初始化
    def __init__(self, estimators, m):
        self.m = m
        self.estimators = estimators
        self.samples = self.random_samples_n(train_df, self.m, len(estimators))

    # 训练模型
    def fit(self, train_df):
        self.train_df = train_df

    # 预测
    def predict(self, test_df, method):
        self.test_df = test_df
        self.method = method
        # 预测结果列表
        prediction_list = []
        # 对各分类器进行训练预测
        for t in range(len(self.estimators)):
            clf_i = self.estimators[t][1]
            train_xi = self.samples[t].drop('cuisine', axis=1).drop('id', axis=1)
            train_yi = self.samples[t]["cuisine"]
            xi_train, xi_valid, yi_train, yi_valid = train_test_split(train_xi, train_yi, test_size=0.2, random_state=42)
            clf_i.fit(xi_train, yi_train)
            prediction_list.append(clf_i.predict(self.test_df))
        if self.method == 'hard':
            self.result_prediction = self.hard_voting(prediction_list=prediction_list)
        else:
            pass
        return self.result_prediction


# 实现bagging
# 创建分类器
clf_lr = LogisticRegression()
clf_knn = KNeighborsClassifier()
clf_df = DecisionTreeClassifier()
clf_svm = SVC(probability=True,)
estimators = [('knn', clf_knn), ('df', clf_df), ('lr', clf_lr), ('svc', clf_svm)]
clf_bag = BaggingClassifier(estimators=estimators, m=0.8)
clf_bag.fit(train_df=train_df)
# 精确度计算并进行对比
for clf in estimators:
    clf[1].fit(x_train, y_train)
    print(fr"Accuracy of {clf[1].__class__.__name__} on training set is {accuracy_metric(y_train, clf[1].predict(x_train))}")
    print(fr"Accuracy of {clf[1].__class__.__name__} on test set is {accuracy_metric(y_valid, clf[1].predict(x_valid))}")
print(fr"Accuracy of bagging on training set is: {accuracy_metric(y_train, clf_bag.predict(test_df=x_train, method='hard').drop('id', axis=1).__getitem__('cuisine'))}")
print(fr"Accuracy of bagging on test set is: {accuracy_metric(y_valid, clf_bag.predict(test_df=x_valid, method='hard').drop('id', axis=1).__getitem__('cuisine'))}")
# 预测
bag_prediction = clf_bag.predict(test_df=test_data, method='hard')
bag_prediction.to_csv(r"bagging_submission.csv", index=False)

