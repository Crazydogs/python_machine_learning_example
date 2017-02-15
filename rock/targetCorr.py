#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 绘制分类数值的相关系交会图

import pandas as pd                     # 一个用于数据处理的库
from pandas import DataFrame            # pandas 中的一种数据结构，表达一个表格结构
import matplotlib.pyplot as plot        # 绘图

from random import uniform

# 读取 csv 格式的数据
dataFile = open('data.txt')
rocksVMines = pd.read_csv(dataFile, header=None, prefix='V')

target = []

for i in range(208):
    if rocksVMines.iat[i, 60] == 'M':
        target.append(1.0 + uniform(-0.1, 0.1))
    else:
        target.append(0.0 + uniform(-0.1, 0.1))

dataRow = rocksVMines.iloc[0:208, 35]
plot.scatter(dataRow, target, alpha=0.5, s=120)
plot.show()
