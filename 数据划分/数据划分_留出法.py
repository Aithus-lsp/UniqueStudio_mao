#!/usr/bin/env python3.8
"""
留出法．

7:3.....．
"""
import pandas as pd 


data = pd.read_csv("train.csv")
length = len(data)
train_data = data[:int(length * 0.7)]
test_data = data[int(length * 0.7):]