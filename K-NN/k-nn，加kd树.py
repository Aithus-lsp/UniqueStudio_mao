#!/usr/bin/env python3.8
"""一个K-NN算法,已实现kd树构造.

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
#处理缺失值
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
#不必处理异常值


#模型函数
def kd_tree(num=0, tree={}, data=train_data):
    """kd树,这又是一个递归算法．

    num: 用于划分点选择的标准．
    tree: 树或子树．
    data: 需要划分的数据．
    """
    #或许对one-hot处理的数据略显奇怪，但还是使用．
    #到底层后返回
    if len(data) < 2:
        tree = data
    else:
        #建立临时列
        feature = data.drop("Survived").columns[num % len(data.columns)]
        line = list(data[feature])
        #取中值
        mid = line.sort()[int(len(line) / 2 - 1)]
        #向下建立字典
        tree[(feature, mid)] = {}
        #建立区分数据集
        divided_data = [data[data[feature] <= mid], data[data[feature] > mid]]
        #防止数据覆盖而使用循环
        for x in range(2):
            feature_copy = copy.deepcopy(feature)
            kd_tree(num + 1, tree[(feature_copy, mid)], data[data[feature_copy] <= mid])
    #返回树
    return tree

def kd_tree_go():

def K_NN(sample, n=7, data=train_data):
    """KNN算法．

    data: 训练数据．
    n: 最近元素数．
    sample: 实例．
    """
    #生成（距离，结果）的序列
    result_line = [(np.sum((data.drop("Survived", axis=1).loc[x])**2 - sample**2), data["Survived"][x])
                    for x in data.index]
    #排序取值
    result_line = result_line.sort()[:v]
    #生成第二个元素(结果)的序列
    out_line = [x[1] for x in result_line]
    #比较，输出
    return [0, 1][out_line.count(0) <= out_line.count(1)]