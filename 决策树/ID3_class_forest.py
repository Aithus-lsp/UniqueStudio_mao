"""由元类继承的ID3随机森林。

这是对随机森林模板和树型ID3的缝合。
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
#分箱,都分为100类，输出为data_cut。
data_cut = data
for x in data_cut: 
    if x != "Outcome":
        data_cut[x] = pd.cut(data_cut[x], 20)
#划分训练、测试数据（本应假设测试数据的特征范围包含于训练据，但难以达到）
train_data = data_cut
test_data = data[int(len(data_cut) * 0.9):]


#定义一些通用函数
#大函数下的小函数，亦可用于计算loss
def cal_entropy_values(series):
    """计算value的熵,以e为底,使用极大似然估计,输入为一个series,输出熵(关于series的值的)"""
    val_count =pd.value_counts(series)
    val_count = val_count[val_count != 0]
    entropy = np.sum(np.log(val_count / np.sum(val_count)) * val_count) / np.sum(val_count)
    return -entropy

def cal_entropy_index(series):
    """计算索引的熵。输入series，输出熵。主要是在计算信息增益比时使用。"""
    index_count = series.index_count()
    entropy = np.sum(np.log(index_count / np.sum(index_count)) * index_count / np.sum(index_count))
    return -entropy

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
    #得到结果,减反了，直接在结果取反
    entropy = np.sum(group_entr * group_count) / np.sum(group_count) - entropy_val
    return -entropy

def Choose_DataFrame(data_dataframe):
    """计算信息增益

    输入一个dataframe(包含outcome),输出最佳特征及其信息增益。
    """
    #生成记录字典
    dic = {}
    #取出outcome
    Outcome_val = data_dataframe["Outcome"].values
    #去除outcome
    data_dataframe_cut = data_dataframe.drop("Outcome", axis=1)
    #填充字典
    for x in data_dataframe_cut:
        #生成临时sreies
        new_ser = pd.Series(Outcome_val, data_dataframe_cut[x].values)
        entropy = Choose_series(new_ser)
        dic[x] = entropy
    #由字典取出最佳特征及其信息增益
    #还是用series吧
    #生成临时series,特征对信息增益
    new_ser_1 = pd.Series(dic)
    #取出索引
    index_1 = new_ser_1.argmax()
    feature = new_ser_1.index[index_1]
    #定义字典
    sub_dict = {}
    for x in data_dataframe[feature]:
        sub_dict[x] = data_dataframe[data_dataframe[feature] == x].drop(feature, axis=1)
    #输出最佳特征子字典及信息增益
    return (feature, sub_dict, new_ser_1[index_1])


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



    def Paint_forest():
        for x in self.forest:
            self.SubPaint(x)
        self.pic.view()

    def Build_forest(self, data2built):
        """随机森林构建函数，以去一个特征为方法。"""
        #森林记录列表
        lis = []
        for x in data2built.drop("Outcome"):
            minidata = data2built.drop(x)
            lis.append((self.SubBuild(minidata))
        self.forest = lis 
    
    def GO_forset(self, INindex):
        """随机森林索引函数"""
        #结果记录
        lis = []
        for x in self.forest:
            index_copy = copy.deepcopy(INindex)
            index_copy = index_copy[]
            lis.append(self.SubGO(INindex, x)[0])
        #利用series输出
        series = pd.Series(lis)
        return series.value_counts.argmax()


#继承
class ID3(DecisionTree):
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
    id3tree = ID3(Choose_DataFrame, cal_entropy_values)
    id3tree.Build_forest(train_data)
    print(id3tree.tree)
    q = test_data.loc[1].drop("Outcome")
    id3tree.GO_forset(q)
    id3tree.Paint_forest()
