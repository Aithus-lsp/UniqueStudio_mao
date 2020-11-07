#!/usr/bin/env python3.8
"""
交叉验证．

使用留一交叉验证
"""
import pandas as pd 


data = pd.read_csv("train.csv")


for x in data.index:
    test_data = data.loc[x]
    train_data = data.drop(x)