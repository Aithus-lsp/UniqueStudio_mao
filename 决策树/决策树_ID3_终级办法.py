#!/usr/bin/env python3\
"""以面向对象为主的ID3构建。

这是一颗通用的树，不限于diabetes.csv数据集。
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  


#开始导入数据,以outcome为输出
data = pd.read_csv("./diabetes.csv")


#处理数据缺失值，但好像没有啊。生成去Outcome数据集(用于索引生成),索引,数据范围,将数据转换为范围(分箱，主要是为了使离散型数据符合要求)
data.dropna(inplace=True)
data_2 = data.drop("Outcome",axis=1)
data_3 = data_2
index_2 = data_2.columns
data_max = data_2.max(axis=0)
data_min = data_2.min(axis=0)
#分箱
for x in data_3:
    data_3[x] = pd.cut(data_3[x], 20)



#定义一些通用函数
def cal_entropy(series):
    """计算value的熵,以e为底,使用极大似然估计,输入为一个series,输出熵(关于series的值的)。
    """
    val_count = series.value_counts()
    entropy = np.sum(np.log(val_count) * val_count) / np.sum(val_count)
    return entropy

def cal_entropy_index(series):
    """计算索引的熵。输入series，输出熵。

    主要是在计算信息增益比时使用。
    """
    index_count = series.index_count()
    entropy = np.sum(np.log(index_count) * index_count / np.sum(index_count))
    return entropy

def Choose_series(data_series):
    """使用信息增益。

    输入series,相应索引及对应outcome,输出其信息增益(以索引为特征，值为输出)。
    """ 
    #以特征不同取值分类
    new_group = data_series.groupby(level=0)
    #计算先验熵
    cal_entropy(data_series)
    #计算每一特征值的熵
    group_entr = new_group.agg(cal_entropy)
    #计数
    group_count = new_group.count()
    #得到结果
    entropy = np.sum(group_entr.values / group_count.values)
    return entropy

def Choose_DataFrame(data_dataframe):
    """计算信息增益

    输入一个dataframe(包含outcome),输出最佳特征及其信息增益
    """
    #生成记录字典
    dic = {}
    #取出outcome
    Outcome_val = data_dataframe["Outcome"].values
    #去除outcome
    data_dataframe.drop("Outcome", axis=1, inplace=True)
    #填充字典
    for x in data_dataframe:
        #生成临时sreies
        new_ser = pd.Series(Outcome_val, data_dataframe[x].values)
        entropy = cal_entropy(new_ser)
        dic[x] = entropy
    #由字典取出最佳特征及其信息增益
    #还是用series吧
    #生成临时series
    new_ser_1 = pd.Series(dic)
    #取出索引
    index_1 = new_ser_1.argmax()
    #输出最佳特征及其信息增益
    return (new_ser_1.index[index_1], new_ser_1[index_1])

#选择特征
def Choose_feature(data_fra):
    """输入一个dataframe(已开始建立索引),返回特征及其信息增益，因为减去值相同
    """
    #生成一个临时group
    grouped = data_fra.groupby(by=index)
    #生成一个记录所有熵的dataframe
    entr_data = grouped.agg(cal_entropy)
    entr_data_out = entr_data.out
    
    

    #生成一个临时series储存数据，保持原顺序


    new_series = pd.Series(out_val, data_series.values)
        #以特征不同取值分类
        new_group = new_series.groupby(level=0)
        #计算熵
        group_entr = new_group.agg(cal_entropy)
        #计数
        group_count = new_group.count()
        entropy = np.sum(group_entr.values / group_count.values)
        bene_dict[x] = entropy

class ID3(object):
    def __init__(self, real_tree={}):
    """一棵树的基本要素。
    realtree: 储存树
    """
        self.real_tree = real_tree

    def BuildTree(self):
        self.realtree = data.set_index(list(index_2))


    def GO(self, attr):
        """索引方法。输入一个包含特征的series,或是一个字典、已相应顺序排列的列表或元组,输出结果。"""
        #转换字典
        if isinstance(attr, dict):
            attr = pd.Series(attr)
        #转换列表，元组
        if isinstance(attr, (list, tuple)):
            attr = pd.Series(attr, index = index_2)
        #判别是否符合索引要求
        if not isinstance(attr, pd.core.series.Series):
            raise TypeError("Only series, tuple, dict, list is allowed")
        #判断数据是否在相应范围内
        if (attr < data_min).all or (attr > data_max).all:
            raise ValueError("values out of range")
        magic_index = list(attr.values)
        #返回相应的outcome
        return self.realtree.loc[magic_index].iloc[-1,0]
        
        