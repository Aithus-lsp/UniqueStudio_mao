#!/usr/bin/env python3
"""真的要用class呀( ⊙ o ⊙ )！

这是一颗决策树，我在好久之前就在图书馆尝试构造，纯粹瞎摸，最后无功而返。
注释待续
"""


import pandas as pd 


#开始导入数据
data = pd.read_csv("./diabetes.csv")


class DecisionTree(object):
    def __init__(self, algor=None, Thresh=(0.01,10), threshVal=0.5):
        self.algor=


#测试
if __name__ == "__main__":
    print(data)