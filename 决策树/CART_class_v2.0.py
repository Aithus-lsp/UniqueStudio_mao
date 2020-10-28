#!/usr/bin/env python3.8
"""
一棵CART，以下内容许多复制于ID3_究极办法与original_class

运行时会警告除零，但没影响。
!!!二分类树好像不用分箱。
这个数据集就运算上来说偏大(我的小破机跑了半分钟)，但是在数据分析上事实上偏小。
这个画图好像有大问题．
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
import copy
from graphviz import Digraph


#开始导入数据,以outcome为输出
data = pd.read_csv("./diabetes.csv")


#处理数据。
#处理数据缺失值。
data.dropna(inplace=True)
#划分训练、测试数据
train_data = data
test_data = data[int(len(data) * 0.9):]


#定义通用函数
#基尼指数
def GINI(data_series):
    """输入一个series,输出基尼指数"""
    val_count = data_series.value_counts()
    gini = 1 - np.sum((val_count / np.sum(val_count))**2)
    return gini

#特征节点选择
def ChooseNode(data_dataframe):
    """输入dataframe，输出特征、划分结果、基尼指数"""
    for x in data_dataframe.drop("Outcome", axis=1):
        #生成记录节点及其最大基尼指数的字典
        dic_gini = {}
        data_copy = copy.deepcopy(data_dataframe)
        new_ser = data_copy.set_index(x)["Outcome"]
        #排序
        new_ser = new_ser.sort_index()
        #以序号为节点，忽略其产生误差
        #生成临时计数字典
        dic={}
        #n为序号而不是目录
        for n in range(len(new_ser)):
            data_1 = new_ser[:n]
            data_2 = new_ser[n:]
            gini = (GINI(data_1) * n + GINI(data_2) * (len(new_ser) - n)) / len(new_ser)
            #字典以具体值为key    
            dic[new_ser.index[n]] = gini
        #用series求解
        ser1 = pd.Series(dic)
        #建立特征指向基尼指数、节点的字典
        dic_gini[x] = (ser1[ser1.argmax()], ser1.argmax())                                                                                                                                                                                                                                                                                                                                      
    #再用series
    ser2 = pd.Series(dic_gini)
    ser2 = ser2.sort_values()
    feature = ser2.index[0]
    gini_result = ser2[feature]
    node = ser2[0][1]
    feature_max = np.max(data_dataframe[feature])
    feature_min = np.min(data_dataframe[feature])
    dic_result = {}
    dic_result[pd.Interval(feature_min, node, closed="right")] = data_dataframe[data_dataframe[feature] <= feature]
    dic_result[pd.Interval(node, feature_max, closed="right")] = data_dataframe[data_dataframe[feature] > feature]
    return (feature, dic_result, gini_result)


class DecisionTree(object):
    """原始树，定义了一些交叉的方法。

    SubBuild: Build子方法，建立树，树的叶结点包含输出和损失。
    SetIndex: Build子方法，建立剪枝时需要的索引，包含深度。
    Build: 调用上二者，保持其递归能力。
    SubGO: GO子方法，为保持递归而定义。
    GO： 索引属性，输入特征输出结果。
    Paint: 可视化
    注： 未定义剪枝，打算在子类中实现。
    """
    def __init__(
        self, choose, cal_loss, fazhi_long=2, fazhi_bene=0.00000000001, 
        index=[], pic=Digraph(comment='DecisonTree')
    ):
        """开始便输入特征选择函数和剪枝函数。

        choose: 输入dataframe,输出特征，特征划分后的结果(dict），特征选择函数值（用于与剪枝）
        cal_loss: 计算叶结点损失函数，未包含正则化。
        fazhi_long: 样本数阀值
        fazhi_bene: 特征选取函数阀值
        index: 训练数据基础上加上索引深度，用于剪枝时向上向下选择,是一个series的列表,深度可用len
        pic: 可视化处理原图
        """
        self.choose = choose
        self.cal_loss = cal_loss
        self.fazhi_long = fazhi_long
        self.fazhi_bene = fazhi_bene
        self.index = index
        self.index_copy = index
        self.pic = pic
    
    def SubBuild(self, INdata):
        """生长函数，提取出来是为方便附加属性。递归函数，输入数据集（dataframe),输出树（字典树）(树中叶结点包含输出值，叶结点损失函数"""
        #预剪枝
        if isinstance(INdata, pd.core.series.Series):
            counted = INdata.value_counts()
            return [counted.argmax(), self.cal_loss(counted)]
        #如果均属于一类，停止并返回，损失为0
        if len(INdata[INdata.columns[-1]].unique()) <=1:
            return [INdata[INdata.columns[-1]].unique()[0], 0]
        #如果样本数小于阀值（主要用于CART),返回结果、损失
        if len(INdata) <= self.fazhi_long:
            counted = INdata["Outcome"].value_counts()
            return [counted.argmax(), self.cal_loss(counted)]
        #特征选择函数调用
        best_feature, sub_dict, bene = self.choose(INdata)
        #如果特征选择函数返回值（如信息增益、基尼指数）小于阀值，停止并返回
        if bene <= self.fazhi_bene:
            counted = INdata[INdata.columns[-1]].value_counts()
            return [counted.argmax(), self.cal_loss(counted)]
        #原始树与添加树
        tree = {best_feature:{}}
        for x in sub_dict:
            tree[best_feature][x] = self.SubBuild(sub_dict[x])
        return tree

    def SetIndex(self, INtree, decide_series=pd.Series(1).drop(0)):
        """建立剪枝时使用的索引
        
        INtree: 输入用于建成索引的树
        decide_series: 最后用于建立深度索引的series(字典的key),因为开始时不能为空，故先建立一个带0索引的series，后再drop
        """
        #到终点（非字典），停止交上
        if not isinstance(INtree, dict):
            decide_series["result"] = INtree
            self.index.append(decide_series)
            return None
        key = list(INtree.keys())[0]
        for x in INtree[key]:
            #建立子series,添加属性
            sub_decide_series = copy.deepcopy(decide_series)
            sub_decide_series[key] = x 
            self.SetIndex(INtree[key][x], sub_decide_series)

    #对self.index预处理文件，使其分包
    def SortIndex(self, index ,count=0):
        """

        index: 需要计数处理的序列
        count: 计数便于截止
        """
        #最开始使第一个文件为列表
        if count == 0:
            index[0] = [index[0]]
            return self.SortIndex(index, count + 1)
        #如果到底，停止处理并返回
        if len(index) < count + 1:
            return index
        #如果此项属于上一序列
        if len(index[count - 1][0]) == len(index[count]):
            #去项并添加至上一序列
            index[count - 1].append(index.pop(count))
            return self.SortIndex(index, count)
        #如果此项不属于上一项
        else:
            index[count] = [index[count]]
            return self.SortIndex(index, count + 1)
        

    def Build(self, data2built):
        """表面调用函数，进行赋值,建立剪枝时索引"""
        self.tree = self.SubBuild(data2built)
        self.SetIndex(self.tree)
        self.index_copy = self.index 
        self.SortIndex(self.index)

    def SubGO(self, INindex, tree):
        """输入特征，树，输出结果(包含损失函数），GO的子方法"""
        #到终点，返回值
        if not isinstance(tree, dict):
            #因为使用tree保存损失函数，故取第一个
            return tree
        #建立一个临时DataFrame
        frame = pd.DataFrame(tree)
        sontree = frame.loc[INindex[tree.keys()][0]][0]
        sonINindex = INindex.drop(tree.keys())
        #实行递归
        return self.SubGO(sonINindex, sontree)
    
    def GO(self, INindex):
        """调用SubGO"""
        return self.SubGO(INindex, self.tree)[0]
    
    def SubPaint(self, tree, cal=0):
        """可视化处理子函数，负责递归"""
        #最开始时初始化
        if cal == 0:
            self.pic.node("-1", "DeciisonTree")
        #停止画图
        if not isinstance(tree, dict):
            self.pic.node(str(cal), str(tree))
            self.pic.edge(str(cal), str(cal - 1))
            return None
        key = list(tree.keys())[0]
        #使用@防止数字带来混合
        self.pic.node(str(cal),str(key))
        self.pic.edge(str(cal-1), str(cal))
        #原发散点序号
        origin = copy.deepcopy(cal)
        for n in tree[key]:
            cal += 1
            self.pic.node(str(cal), str(n))
            self.pic.edge(str(origin), str(cal))
            self.SubPaint(tree[key][n], cal + 1)
    
    def Paint(self):
        self.SubPaint(self.tree)
        self.pic.view()


#继承
#继承
class CART(DecisionTree):
    """继承于Decision的ID3"""
    def Cut(self, para=0.1):
        """定义剪枝函数。

        para： 正则化参数
        """
        #只计算损失函数数值改变正负
        pass
        for x in range():
            newlist = [n for n in self.index if len(n[0]) == x]
            for a in newlist:
                #计算正则化改变量
                cal_zhenze = (len(a) - 1) * para
                #单纯的损失函数改变
                #原结果列表
                lis_1 = [n["result"][0] for n in a]
                #原损失函数列表
                lis_2 = [n["result"][1] for n in a]
                #原来损失
                list_loss = np.sum(lis_2)
                #后来损失
                new_outcome = pd.Series(lis_1).value_counts.argmax()
                new_loss = 0
                for b in range(len(lis_1)):
                    if lis_1[b] == new_outcome:
                        new_loss += 1

                loss_chang = 1

                return

            


    pass


if __name__ == "__main__":
    #使用
    carttree = CART(ChooseNode, GINI)
    carttree.Build(train_data)
    print(carttree.tree)
    q = test_data.loc[700].drop("Outcome")
    carttree.GO(q)
    carttree.Paint()