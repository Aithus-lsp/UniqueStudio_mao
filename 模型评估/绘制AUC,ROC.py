#!/usr/bin/env python3.8
"""
使用AUC,ROC绘制的原型．

对逻辑回归绘制，但sigmoid越阶函数不适合画这种图．
我假设的数据是一个二分类且真实结果以一个０１列表排列，预测列表以可能为１顺序排列（二者等长,就实例而言二者等序）．
"""
import matplotlib.pyplot as plt



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
    FRP = []
    for x in range(len(line_predict)):
        new_line = [1] * (x + 1) +[0] * (len(line_predict) - x -1)
        #取真值
        FP = sum([new_line[y] != line_real[y] for y in range(len(new_line)) if new_line[y] == 1])
        #取真实中的真的数量
        total_N = line_real.count(0)
        FRP.append(FP / total_N)
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