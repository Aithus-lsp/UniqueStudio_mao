#!/usr/bin/env python3
"""真的要用class呀( ⊙ o ⊙ )！

这是关于决策树尝试实验的文档，打算先写ID3，写其它树时会从这里import,但是这里的注释不会使用模块的标准注释。
吐槽一下，我在好久之前就在图书馆尝试构造，纯粹瞎摸，最后无功而返。
这次在定义类的时候直接将数据使用。
注释待续
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  


#开始导入数据,以outcome为输出
data = pd.read_csv("./diabetes.csv")


#处理数据缺失值，但好像没有啊。
data.dropna(inplace=True)


#害怕出错，定义一些通用函数
#计算熵,以e为底,使用极大似然估计,输入为一个series
    def cal_entropy(series):
        val_count = series.value_counts()
        entropy = np.sum(np.log(val_count) * val_count) / np.sum(val_count)
        return entropy


class DecisionTree(object):
"""一颗通用决策树。
不对特征进行选择，丢弃。
它将保存一些基础属性，函数，有：
"""

    
    def __init__(self, algor=None, Thresh=(0.01,10), ThreshVal=0.5):
        """一棵树的基本要素，分别为阀值,"""
        self.__algor = 
        self.__Thresh =
        self.__ThreshVal =

    
    #寻找分界点，使用信息增益比
    
    #输入数据计算信息增益
    bene_dict={}
    #计算信息熵
    out_bene = cal_entropy(data.Outcome)
    #生成信息增益字典
    for x in data: 
        new_series = pd.Series(data["Outcome"].values, data[x].values)
        new_group = new_series.groupby(x)
        group_entr = new_group.agg(cal_entropy)
        group_count = new_group.count()
        entropy = np.sum(group_entr.values / group_count.values)
        bene_dict[x] = entropy
    
    #生成

    
        
        
    def ChooseBest(self, ):
        #使用信息增益比
        

    #建造树
    def Buildtree(self, ):

    #剪枝
    def REPrunning(self, ):

    #树的可视化
    def Painttree(self, ):


#一颗ID3
class ID3(DecisionTree):
    def __init__(self, algor="ID3", Thresh=(0.0001, 8), threshVal=0.5):
        super(ID3, self).__init__()
        self.__algor = algor
        self.__Thresh = Thresh
        self.__threshVal = threshVal


#测试
if __name__ == "__main__":
    print(data)
    Tree = ID3()
    Tree.Buildtree