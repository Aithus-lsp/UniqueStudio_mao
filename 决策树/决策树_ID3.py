#!/usr/bin/env python3.8
"""真的要用class呀( ⊙ o ⊙ )！

这是关于决策树尝试实验的文档，打算先写ID3，写其它树时会从这里import,但是这里的注释不会使用模块的标准注释。
吐槽一下，我在好久之前就在图书馆尝试构造，纯粹瞎摸，最后无功而返。
这里，我将生成一个多维列表。
ｅｍｍ，所以关键还是索引吗？那直接用函数不好吗？还是说为了树的继承。以多个索引取代单个索引，那样的话cart干脆使用布尔索引好了。
这索引好像numpy的神奇索引啊。
关键是索引的应用，树应该长什么样子呢？(逐渐远离注释......)
这次在定义类的时候直接将数据使用。
注释待续
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  


#开始导入数据,以outcome为输出
data = pd.read_csv("./diabetes.csv")


#处理数据缺失值，但好像没有啊。生成去Output数据集
data.dropna(inplace=True)
data_2 = data.drop("Outcome",axis=1)


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

    def __init__(self, algor=None, fazhi=3, Thresh=(0.01,10), ThreshVal=0.5, real_tree):
        """一棵树的基本要素。

        algor:　树的名字
        fazhi:　分枝的最小阀值，信息增益比小于之则停止分枝
        Thresh:
        real_tree: 真正构造
        """
        self.__algor = algor
        self.__fazhi = fazhi
        self.__Thresh = Thresh
        self.__ThreshVal = ThreshVal

    def ChooseBest(self, data_set):
        #使用信息增益比
        bene_dict={}
        #计算信息熵
        out_bene = cal_entropy(data_set.Outcome)
        #生成信息增益字典
        for x in data_set: 
            new_series = pd.Series(data_set["Outcome"].values, data_set[x].values)
            #以特征不同取值分类
            new_group = new_series.groupby(level=0)
            #计算
            group_entr = new_group.agg(cal_entropy)
            group_count = new_group.count()
            entropy = np.sum(group_entr.values / group_count.values)
            bene_dict[x] = entropy
        #使用series取特征
        ser_1 = pd.Series(bene_dict)
        index = ser_1.index[ser_1.argmax]
        bene = bene_dict[index]
        #返回特征及其信息增益比
        return (index, bene)

        
    #剪枝
    def REPrunning(self, ):

    #树的可视化
    def Painttree(self, ):


#一颗ID3
class ID3(DecisionTree, pd.core.frame.DataFrame):
    """一颗ID3.
    将构建ID3特有的索引,啊,直接从dataframe中偷不就好了。
    多重索引和列表索引都要！！！
    """
    
    def __init__(self, algor="ID3", Thresh=(0.0001, 8), threshVal=0.5, deep = 0, data_stored = data_2):
    """原始函数

    deep: 记录节点深度
    data_stored: 在造树过程中对特征删减，保持原数据集完整性
    super(ID3, self).__init__()
    self.__algor = algor
    self.__Thresh = Thresh
    self.__threshVal = threshVal
    self.data = data_stored
    """
    
    #建造树,还是递归好用，函数转移特定算法下构建,就不构建一些特殊情况的函数了！
    #开始时尝试在已建成基础上修改，后打算先建立索引再依据此构建
    def Buildtree(self, data_set):
        node_choosed = ChooseBest(data_set)
        if node_choosed[1] < fazhi:
            continue
        node = list(data_set[node_choosed(0)].unique())
        self.append()
        self.data.drop(node_choosed[0], inplace=True)

#测试
if __name__ == "__main__":
    print(data)
    Tree = ID3()
    Tree.Buildtree()