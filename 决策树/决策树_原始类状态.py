#!/usr/bin/env python3
"""原始DecisionTree万树之母，其它树均由此开始。

此文件仅作为提取模块，不进行数据处理等。
"""
class DecisionTree(object):
    def __init__(self, cal_choose, cal_tree, tr_data):
        """输入基本要素。
        cal_choose: 选择特征的函数。
        cal_tree: 建树函数，因节点生成函数不同，难以同化。
        tr_data: 生成树的数据集。
        """
        self.cal_choose = cal_choose
        self.cal_tree = cal_tree
        self.tr_data = tr_data
    def ChooseBest(self):
        """选择特征的方法，输出特征及用于选择指数"""
        return self.cal_choose(self.tr_data)
    def BuildTree()