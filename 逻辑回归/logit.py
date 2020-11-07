#!/usr/bin/env python3.8
"""
逻辑斯谛回归．

使用sigmoid函数
"""

#导入模块
import numpy as np
import pandas as pd 
import copy
import matplotlib.pyplot as plt


#导入处理数据
#导入
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
test_result = pd.read_csv("gender_submission.csv")
#处理
#去除名字,票号
train_data.drop(["Name", "Ticket"], axis=1, inplace=True)
test_data.drop(["Name", "Ticket"], axis=1, inplace=True)
#以passengerid为序列,方便结果核对
train_data.set_index("PassengerId", inplace=True)
test_data.set_index("PassengerId", inplace=True)
test_result.set_index("PassengerId", inplace=True)
#Cabin缺失的数据过多,不好丢弃,填充loss
train_data.Cabin.fillna("loss", inplace=True)
test_data.Cabin.fillna("loss", inplace=True)
#处理缺失值
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
#不必处理异常值
#one-hot处理
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)
#删除train_data,test_data中多余的列,取交集
inter_col = train_data.columns.intersection(test_data.columns)
train_data_in = train_data[inter_col]
train_data_in["Survived"] = train_data["Survived"]
train_data = train_data_in
test_data = test_data[inter_col]
#取预测结果中的相应行
test_result = test_result.loc[test_data.index]


#预选数据
#权值，包含偏置
w = np.zeros(44)
#造函数
def sigmoid(x):
    """sigmoid函数，用于二分类决策"""
    return np.exp(x) / (1 + np.exp(x))

def train_logit(data_output=train_data["Survived"], data_droped=copy.deepcopy(train_data.drop("Survived", axis=1))
                , digit=0.001, step=0.001):
    """模型训练函数．
    
    data_output: 训练数据中的输出．
    data_droped: 删除输出的训练数据．
    digit: 精度，决定半路停止．
    step: 步长，懒得在算法中找了．
    """
    global w
    #在训练数据上加上用于偏置的一行
    data_droped["added"] = 1
    #开始有限次回归．
    for x in range(4000):
        #输出为一个series,每个id对损失
        loss = sigmoid(np.dot(data_droped, w) - data_output)
        if np.sum(loss**2) < digit:
            return
        #计算梯度
        rate = np.array([np.sum(data_droped[x]* loss) for x in data_droped])
        w += step * rate

def logit(sample):
    """利用模型对数据预测
    
    sample: 预测样本．
    """
    sample_copy = copy.deepcopy(sample)
    sample_copy["added"] = 1
    #生成判断对错
    judged = sigmoid(np.dot(sample_copy, w)) > 0.5
    #转换为０１型
    sur = []
    for x in judged:
        if x:
            sur.append(1)
        else:
            sur.append(0)
    return np.array(sur)

#画图函数
def cal_TPR(line_predict, line_real):
    """计算一个真正例率列表．

    line_predict: 预测列表．
    line_real: 真实列表．
    """
    TPR = []
    for x in range(len(line_predict)):
        new_line = [1] * (x + 1) +[0] * (len(line_predict) - x -1)
        #取真值
        TP = sum([new_line[y] == line_real[y] for y in range(len(new_line)) if new_line[y] == 1])
        #取真实中的真的数量
        total_P = line_real.count(1)
        TPR.append(TP / total_P)
    return TPR

def cal_FPR(line_predict, line_real):
    """计算一个假正例率列表．

    line_predict: 预测列表．
    line_real: 真实列表．
    """
    FPR = []
    for x in range(len(line_predict)):
        new_line = [1] * (x + 1) +[0] * (len(line_predict) - x -1)
        #取真值
        FP = sum([new_line[y] != line_real[y] for y in range(len(new_line)) if new_line[y] == 1])
        #取真实中的真的数量
        total_N = line_real.count(0)
        FPR.append(FP / total_N)
    return FPR
    
def ROC(line_predict, line_real):
    """绘制ROC曲线

    line_predict: 预测列表．
    line_real: 真实列表．
    """
    #计算xy轴
    x = cal_FPR(line_predict, line_real)
    y = cal_TPR(line_predict, line_real)
    plt.figure()
    plt.plot(x, y)

def AUC(line_predict, line_real):
    """绘制AUC曲线

    line_predict: 预测列表．
    line_real: 真实列表．
    """
    #计算轴
    x_1 = cal_FPR(line_predict, line_real)
    y_1 = cal_TPR(line_predict, line_real)
    AUC_val = 0
    for x in range(len(x_1) -1):
        AUC_val += (x_1[x + 1] - x_1[x]) * (y_1[x + 1] + y_1[x])
    return AUC_val / 2


if __name__ == "__main__":
    train_logit()
    test_data["added"] = 1
    result = logit(test_data)
    #计算准确率
    length = len(test_data)
    acc_line = []
    for x in range(len(result)):
        if result[x] == test_result.iloc[x][0]:
            acc_line.append(1)
        else:
            acc_line.append(0)
    percent = sum(acc_line) / length *100
    print("The accuracy is %fpercent"%percent)
    predict = sigmoid(np.dot(test_data, w))
    test_result["pre"] = predict
    test_result.sort_values(by="pre")
    predic = list(test_result["pre"])
    resu = list(test_result["Survived"])
    ROC(predic, resu)
    print(AUC(predic, resu))
    