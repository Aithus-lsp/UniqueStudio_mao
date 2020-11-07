#!/usr/bin/env python3.8
"""
一个Kd树.

使用欧几里得距离,不对数据进行加权.还没写出索引办法．
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


#先定义一个树
kd_tree = {}


#函数
def mk_kd_tree(deep=0, tree=kd_tree, data_droped=train_data.drop("Survived", axis=1), 
            data=train_data, length=(len(train_data) - 1)):
    """kd树建成算法．

    parameter:
        deep: 深度，用于选择特征．
        tree: 原始树或节点．
        data_droped: 删除Survived后的数据．
        data: 原数据．
        length: 特征长度．
    """
    global kd_tree
    #到节点，返回
    if isinstance(data, pd.core.series.Series) or len(data_droped) == 0:
        return
    #取出特征
    feature = data_droped.columns[deep % length]
    #取出特征列表
    feature_line = list(data_droped[feature])
    #取出特征中值
    mid = sorted(feature_line)[len(feature_line) // 2]
    #取出节点数据
    mid_data = data[data[feature] == mid]
    if isinstance(mid_data, pd.core.frame.DataFrame):
        mid_series = mid_data.iloc[0]
    else:
        mid_series = mid_data
    #取出上数据，下数据,递归建树，可能会有数据的损失
    tree[(mid_series[feature], feature, mid_series["Survived"])] = {}
    #避免数据覆盖使用循环加深拷贝
    for x in range(2):
        mid_copy = copy.deepcopy(mid)
        data_droped_copy = copy.deepcopy(data_droped)
        data_copy = copy.deepcopy(data)
        if x ==0:
            mk_kd_tree(deep + 1, tree[(mid_series[feature], feature, mid_series["Survived"])], data_droped_copy[data_droped_copy[feature] > mid],
                    data_copy[data_copy[feature] > mid_copy], length)
        else:
            mk_kd_tree(deep + 1, tree[(mid_series[feature], feature, mid_series["Survived"])], data_droped_copy[data_droped_copy[feature] < mid],
                    data_copy[data_copy[feature] < mid_copy], length)


#测试
if __name__ == "__main__":
    mk_kd_tree()