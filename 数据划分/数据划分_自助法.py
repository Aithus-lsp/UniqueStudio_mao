#!/usr/bin/env python3.8
"""
自助法．

随机取值．
"""
import pandas as pd
import numpy as np 


data = pd.read_csv("train.csv")
length = len(data)
picked_data = data.loc[int(length * np.random.uniform())]