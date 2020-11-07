#!/usr/bin/env python3.8
"""
一个K-NN算法.

使用欧几里得距离,不对数据进行加权.
"""


#导入模块
import numpy as np
import pandas as pd 
import copy


#导入处理数据
#导入
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
test_result = pd.read_csv("gender_submission.csv")
#处理
#去除名字,票号
train_data.drop(["Name", "Ticket"], axis=1, inplace=True)
test_data.drop(["Name", "Ticket"], axis=1, inplace=True)
#以passengerid为序列,方便结果核对
train_data.set_index("PassengerId", inplace=True)
test_data.set_index("PassengerId", inplace=True)
test_result.set_index("PassengerId", inplace=True)
#Cabin缺失的数据过多,不好丢弃,填充loss
train_data.Cabin.fillna("loss", inplace=True)
test_data.Cabin.fillna("loss", inplace=True)
#处理缺失值
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
#不必处理异常值
#one-hot处理
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)
#删除train_data,test_data中多余的列,取交集
inter_col = train_data.columns.intersection(test_data.columns)
train_data_in = train_data[inter_col]
train_data_in["Survived"] = train_data["Survived"]
train_data = train_data_in
test_data = test_data[inter_col]
#取预测结果中的相应行
test_result = test_result.loc[test_data.index]


#模型函数
def K_NN(sample=test_data, n=7, data=train_data):
    """KNN算法．

    data: 训练数据．
    n: 最近元素数．
    sample: 实例．
    """
    sur = []
    for y in sample.index:
        #生成（距离，结果）的序列 
        result_line = [(((data.drop("Survived", axis=1).loc[x]) ** 2 - sample.loc[y] ** 2).sum(), data["Survived"][x])
                    for x in data.index]
        #排序取值
        result_line = sorted(result_line)[:n]
        #生成第二个元素(结果)的序列
        out_line = [x[1] for x in result_line]
        if 2 * out_line.count(1) <= n:
            sur.append(1)
        else:
            sur.append(0)
    return np.array(sur)


if __name__ == "__main__":
    result = K_NN()
    #计算准确率
    length = len(test_data)
    acc_line = []
    for x in range(len(result)):
        if result[x] == test_result.iloc[x][0]:
            acc_line.append(1)
        else:
            acc_line.append(0)
    percent = sum(acc_line) / length *100
    print("The accuracy is %fpercent"%percent)