#!/usr/bin/env python3
"""真的要用class呀( ⊙ o ⊙ )！

这是关于决策树尝试实验的文档，打算先写ID3，写其它树时会从这里import,但是这里的注释不会使用模块的标准注释。
吐槽一下，我在好久之前就在图书馆尝试构造，纯粹瞎摸，最后无功而返。
注释待续
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  


#开始导入数据,以outcome为输出
data = pd.read_csv("./diabetes.csv")


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

    #寻找分界点
    #计算熵,以e为底,使用极大似然估计,输入为一个series
    def cal_entropy(self, series):
        entropy = sum([])
        return entropy
    def cal_bene(self):
    """输入数据计算信息增益"""
        bene_dict={}
        out_bene = cal_entropy(data.Outcome)
        for x in data.index:
            bene_dict[x] = cal_entropy(series)
        return bene_dict
        
        
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