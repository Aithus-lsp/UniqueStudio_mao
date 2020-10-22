#!/usr/bin/env python3
"""真的要用class呀( ⊙ o ⊙ )！

这是关于决策树尝试实验的文档，打算先写ID3，写其它树时会从这里import,但是这里的注释不会使用模块的标准注释。
吐槽一下，我在好久之前就在图书馆尝试构造，纯粹瞎摸，最后无功而返。
注释待续
"""


import pandas as pd 
import matplotlib.pyplot as plt 


#开始导入数据
data = pd.read_csv("./diabetes.csv")


class DecisionTree(object):
"""一颗通用决策树。
将保存一些基础属性，函数，有：
"""

    def __init__(self, algor=None, Thresh=(0.01,10), ThreshVal=0.5):
        """一棵树的基本要素，分别为阀值,"""
        self.__algor = 
        self.__Thresh =
        self.__ThreshVal =

    #寻找分界点
    def ChooseBest(self, ):
        #使用gini指数

    #建造树
    def Buildtree(self, ):

    #剪枝
    def REPrunning(self, ):

    #树的可视化
    def Painttree(self, ):


#一颗ID3
class ID3(DecisionTree):


#测试
if __name__ == "__main__":
    print(data)
    tree = 