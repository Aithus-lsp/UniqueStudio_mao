#!/usr/bin/env python3.8
"""
一棵C4.5。
以下内容大都复制于ID3_究极办法

运行时会警告除零，但没影响。
!!!划分训练数据时得到的数据即使分箱后也不能覆盖测试数据的全部。因个人能力，我只好对训练数据进行的部分丢弃。
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
#分箱,都分为20类，输出为data_cut。
data_cut = data
for x in data_cut: 
    if x != "Outcome":
        data_cut[x] = pd.cut(data_cut[x], 20)
#划分训练、测试数据
train_data = data_cut
test_data = data[int(len(data_cut) * 0.9):]


#定义一些通用函数
def cal_entropy_values(series):
    """计算value的熵,以e为底,使用极大似然估计,输入为一个series,输出熵(关于series的值的)"""
    val_count =pd.value_counts(series)
    val_count = val_count[val_count != 0]
    entropy = np.sum(np.log(val_count / np.sum(val_count)) * val_count) / np.sum(val_count)
    return -entropy

def cal_entropy_index(series):
    """计算索引的熵。输入series，输出熵。

    主要是在计算信息增益比时使用。
    """
    #转置目录
    index2val = pd.Series(series.index)
    return cal_entropy_values(index2val)

def Choose_series(data_series):
    """使用信息增益。

    输入series,相应索引及对应outcome,输出其信息增益(以索引为特征，值为输出)。
    会报除零，不要在意。
    """ 
    #以特征不同取值分类
    new_group = data_series.groupby(by=data_series.index)
    #计算先验熵
    entropy_val = cal_entropy_values(data_series)
    #计算每一特征值的熵
    group_entr = new_group.apply(cal_entropy_values).dropna()
    #计数
    group_count = new_group.count()
    group_count = group_count[group_count != 0]
    #取索引的熵
    entropy_index = cal_entropy_index(data_series)
    #得到结果
    entropy = np.sum(group_entr * group_count) / np.sum(group_count) - entropy_val
    return -entropy/entropy_index

def Choose_DataFrame(data_dataframe):
    """计算信息增益

    输入一个dataframe(包含outcome),输出最佳特征及其信息增益。
    """
    #生成记录字典
    dic = {}
    #取出outcome
    Outcome_val = data_dataframe["Outcome"].values
    #去除outcome
    data_dataframe = data_dataframe.drop("Outcome", axis=1)
    #填充字典
    for x in data_dataframe:
        #生成临时sreies
        new_ser = pd.Series(Outcome_val, data_dataframe[x].values)
        entropy = Choose_series(new_ser)
        dic[x] = entropy
    #由字典取出最佳特征及其信息增益
    #还是用series吧
    #生成临时series
    new_ser_1 = pd.Series(dic)
    #取出索引
    index_1 = new_ser_1.argmax()
    #输出最佳特征及其信息增益
    return (new_ser_1.index[index_1], new_ser_1[index_1])


#开始C4.5
def C45(INdata, fazhi=0.0000000000001):
    """C4.5构建函数。

    INdata: 输入数据。
    fazhi: 阀值，当信息增益小于它时停止建树
    """
    
    #停止生长判断
    #如果特征集为零，停止并返回特征
    if isinstance(INdata, pd.core.series.Series):
        count = INdata.value_counts()
        return count.argmax()
    #如果均属于一类，停止并返回
    if len(INdata[INdata.columns[-1]].unique()) <=1:
        return INdata[INdata.columns[-1]].unique()[0]
    #计算信息增益，最佳特征
    best_feature, bene = Choose_DataFrame(INdata)
    #如果信息增益小于阀值，停止并返回
    if bene <= fazhi:
        count = INdata[INdata.columns[-1]].value_counts()
        return count.argmax()
    
    #原始树与添加树
    c45tree={best_feature:{}}
    for x in INdata[best_feature].unique():
        data_droped = INdata[INdata[best_feature] == x]
        sontree = C45(data_droped)
        c45tree[best_feature][x] = sontree
    return c45tree


#树的使用
def GO(INindex, tree):
    """输入一个series,输出预测值"""
    #到终点，返回值
    if not isinstance(tree, dict):
        return tree
        #建立一个临时DataFrame
    frame = pd.DataFrame(tree)
    sontree = frame.loc[INindex[tree.keys()][0]][0]
    sonINindex = INindex.drop(tree.keys())
    #实行递归
    return GO(sonINindex, sontree)


if __name__ == "__main__":
    my_c45tree = C45(train_data)
    a = train_data.loc[0]
    b = GO(a, my_c45tree)
    #计算准确率
    test = pd.Series([GO(test_data.loc[x], my_c45tree) for x in test_data.index])
    print(test)