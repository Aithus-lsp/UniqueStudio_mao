import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
import copy
from graphviz import Digraph

def cal_entropy_values(series):
    """计算value的熵,以e为底,使用极大似然估计,输入为一个series,输出熵(关于series的值的)"""
    val_count =pd.value_counts(series)
    val_count = val_count[val_count != 0]
    entropy = np.sum(np.log(val_count / np.sum(val_count)) * val_count) / np.sum(val_count)
    return -entropy
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


#以上内容均为辅助,不符格式,请无视
def Prun(self, max_deep, tree_input, original_data_input, cal_loss, regu=0.02):
    """剪枝函数.输出剪枝后的树,类内置方法,不用管作用域.

    max_deep: 最大深度
    tree_input:  需要剪枝的树．
    original_data_input: 原始数据
    cal_loss: 计算损失函数
    regu: 正则化参数
    """

    #辅助函数
    def Find_data(feature_series, original_data=original_data_input):
        """对原始数据进行搜索的函数.输出为outcome的series

        feature_series: 需要检索的特征.
        original_data: 原始数据.
        """
        out_line = []
        for x in original_data.index:
            if (data_copyed.loc[x][d2s.index] == d2s).all():
                out_line.append(original_data.loc[x]["Outcome"])
        return pd.Series(out_line)
    def Godeep(tree=tree_input, deepnow=0, save_dict={}, pre_save_dict={}, pre_tree=tree_input, deep=max_deep):
        """Prun内置递归函数.主要调用函数.
        
        deep: 深度规划，保证剪枝由深到浅(或反向)进行．
        deepnow: 现在深度.
        save_dict: 记录特征.
        pre_save_dict: 上一个字典.
        pre_tree:上一个树,方便剪枝.
        """
        #判断是否到底及进行剪枝
        if not isinstance(tree, dict):
            if deepnow == deep:
                #转换dict为series方便搜索
                d2s = pd.Series()
                #取出原数据中的相关outcome
                data_copyed = copy.deepcopy(original_data_input)
                #原始outcome
                ori_out = Find_data(save_dict)
                #剪枝后的outcome
                cut_out = Find_data(pre_save_dict)
                #比较,剪枝,正则化以比较子树数量变化给出
                if (cal_loss(ori_out) + regu * (len(ori_out) -1)) >= cal_loss(cut_out):
                    pre_tree = cut_out.value_counts().argmax()
        else:
            key = list(tree.keys())[0]
            for x in tree[key]:
                #深拷贝防止数据出错
                save_dict_copy = copy.deepcopy(save_dict)
                save_dict_copy[key] = x
                #实行递归,这里按顺序输入
                Godeep(tree[key][x], deepnow + 1, save_dict_copy, save_dict, tree) 
    
    Godeep()
    return tree_input
                    