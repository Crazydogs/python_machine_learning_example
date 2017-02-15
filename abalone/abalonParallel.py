#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 回归问题的折线图

import pandas as pd                     # 一个用于数据处理的库
from pandas import DataFrame            # pandas 中的一种数据结构，表达一个表格结构
import matplotlib.pyplot as plot        # 绘图
from math import exp

dataFile = open('abalone.txt')
abalone = pd.read_csv(dataFile, header=None, prefix='V')
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
    'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

summary = abalone.describe()
minRings = summary.iloc[3, 7]
maxRings = summary.iloc[7, 7]

nrows = len(abalone.index)

for i in range(nrows):
    dataRow = abalone.iloc[i, 1:8]
    # 对数据进行归一化处理
    labelColor = (abalone.iloc[i, 8] - minRings) / (maxRings - minRings)
    dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)

plot.show()

meanRings = summary.iloc[1, 7]
sdRings = summary.iloc[2, 7]

# for i in range(nrows):
#     dataRow = abalone.iloc[i, 1:8]
#     normTarget = (abalone.iloc[1, 8] - meanRings) / sdRings
#     labelColor = 1.0 / (1.0 + exp(-normTarget))
#     dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)
# 
# plot.show()

corMat = DataFrame(abalone.iloc[:, 1:9].corr())
plot.pcolor(corMat)
plot.show()
