#!/usr/bin/env python3
"""突然发现ID3用DataFrame写真的简单

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  


#开始导入数据,以outcome为输出
data = pd.read_csv("./diabetes.csv")


#处理数据缺失值，但好像没有啊。生成去Output数据集(用于索引生成)
data.dropna(inplace=True)
data_2 = data.drop("Outcome",axis=1)


#设定一些常量
#节点深度
deep = 0
#特征数量
long = (len(data.columns) - 1)
#特征集
data_feature=[]


#定义一些通用函数
def cal_entropy(series):
    """计算熵,以e为底,使用极大似然估计,输入为一个series,输出熵(关于series的值的)。
    """
    val_count = series.value_counts()
    entropy = np.sum(np.log(val_count) * val_count) / np.sum(val_count)
    return entropy

def cal_entropy_index(series):
    """计算索引的熵
    """
    index_count = series.index_count()
    entropy = np.sum(np.log(index_count) * index_count / np.sum(index_count))
    return entropy

def Choose_series(data_series):
    """使用信息增益比。
    输入series,相应索引及对应outcome,输出其信息增益比
    """ 
    #以特征不同取值分类
    new_group = data_series.groupby(level=0)
    #计算熵
    cal_entropy(data_series)
    #计算每一特征值的熵
    group_entr = new_group.agg(cal_entropy)
    #计数
    group_count = new_group.count()
    entropy = np.sum(group_entr.values / group_count.values)-
    return entropy

#选择特征
def Choose_feat(data_fra):
    """输入一个dataframe(已开始建立索引),返回特征及其信息增益比，因为减去值相同
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


#建树
def BuildTrees(data_set，fazhi = 1):
    """输入数据(可以是分层或不分层的DataFrame),阀值，在原数据基础上叠加一个分层索引，应该支持迭代
    """
    global deep, data_2, data, data_feature
    #开始选取特征
    #如果未开始生成索引
    if deep == 0:
        node_choosed = ChooseBest(data_2)
        feature = list(data_set[node_choosed(0)].unique())
        #设计离开循环条件
        if node_choosed[1] < fazhi or deep > long:
            print("end")
            break
        data.set_index(node_choosed[0], inplace=True)
        deep += 1
        data_feature.append(feature)
        continue
    #data_feature不为空
    #以已生成数据建立一个临时组，组内容为DataFrame
    grouped = data.groupby(by=data.index)
    #生成一个关于原数据结构相似的DataFrame
    feature_choosed = grouped.agg(ChooseBest)
    
    node_choosed = ChooseBest(data_2)
    if node_choosed[1] < fazhi or deep > long:
        break
    feature = list(data_set[node_choosed(0)].unique())
    data.set_index(node_choosed[0], inplace=True)
    deep += 1
    data_2.drop(node_choosed[0], inplace=True)
    data_feature.append(feature)


#计算信息熵
out_bene = cal_entropy(data.Outcome)
#生成outcome的ndarray
out_val = data["Outcome"].values
for x in range(long)
