#!/usr/bin/env python3.8
"""
一棵CART。
以下内容许多复制于ID3_究极办法

运行时会警告除零，但没影响。
!!!二分类树好像不用分箱。
这个数据集就运算上来说偏大(我的小破机跑了半分钟)，但是在数据分析上事实上偏小。
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  


#开始导入数据,以outcome为输出
data = pd.read_csv("./diabetes.csv")


#处理数据。
#处理数据缺失值。
data.dropna(inplace=True)
#划分训练、测试数据
train_data = data
test_data = data[int(len(data_cut) * 0.9):]


#定义通用函数
#基尼指数
def GINI(data_series):
    """输入一个series,输出基尼指数"""
    val_count = data_series.value_counts()
    gini = 1 - np.sum((val_count / np.sum(val_count))**2)
    return gini

#特征节点选择
def ChooseNode(data_dataframe):
    """输入dataframe，输出特征、节点"""
    for x in data_dataframe.drop("Outcome", axis=1):
        #生成记录节点及其最大基尼指数的字典
        dic_gini = {}
        new_ser = data_dataframe.set_index(x)["Outcome"]
        new_ser = new_ser.sort_index()
        #以序号为节点，忽略其产生误差
        #生成临时计数字典
        dic={}
        #n为序号而不是目录
        for n in len(new_ser):
            data_1 = new_ser[:n]
            data_2 = new_ser[n:]
            gini = (GINI(data_1) * n + GINI(data_2) * (len(new_ser) - n)) / len(new_ser)
            #字典以区间为key    
            dic[new_ser.index[n]] = gini
        #用series求解
        ser1 = pd.Series(dic)
        #建立特征指向基尼指数、节点的字典
        dic_gini[x] = (ser1[ser1.argmax()], ser1.argmax())                                                                                                                                                                                                                                                                                                                                      
    #再用series
    ser2 = pd.Series(dic_gini)
    ser2 = ser2.sort_values()
    return (ser2.index[0], ser2[0][1])

#开始种树
def CART(data, fazhi_sample=2, fazhi_gini=0.0000000000001):
    """输入训练的dataframe,输出树
    fazhi_sample: 样本阀值
    fazhi_gini: 基尼指数阀值
    """
    feature, node = ChooseNode(data)
    if len(data) <= fazhi_sample or gini <= fazhi_gini::
        return data["Outcome"].value_counts().argmax()
    carttree = {feature: {}}
    #处理二分类
    data1 = CART(data[data[feature] <= node])
    data2 = CART(data[data[feature] > node])
    carttree[feature][node] = data1
    carttree[feature][1] = data2
    return carttree

#索引函数
def GO(data_series, tree):
    """输入包含特征的series,输出类别"""
    #到底时输出
    if not isinstance(tree, dict):
        return tree
    subtree = data_series[tree.keys()]
    node = list[subtree.keys()].remove(1)[0]
    #样本特征值
    val = data_series[tree.keys()]
    if val <= node:
        return GO(data_series, subtree[node])
    else:
        return GO(data_series, subtree[1])
#测试
if __name__ == "__main__":
    print(ChooseNode(train_data))