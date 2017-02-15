#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 绘制箱线图

import pandas as pd                     # 一个用于数据处理的库
from pandas import DataFrame            # pandas 中的一种数据结构，表达一个表格结构
import matplotlib.pyplot as plot        # 绘图
from pylab import *

dataFile = open('abalone.data')
abalone = pd.read_csv(dataFile, header=None, prefix='V')

abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
    'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

summary = abalone.describe()
print(summary)

array = abalone.iloc[:, 1:9].values
boxplot(array)
show()

array2 = abalone.iloc[:, 1:8].values
boxplot(array2)
show()

# 归一化处理，防止线箱图各项差异过大
abaloneNormalized = abalone.iloc[:, 1:9]
for i in range(8):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]

    abaloneNormalized.iloc[:, i:(i + 1)] = (
        abaloneNormalized.iloc[:, i:(i + 1)] - mean) /sd

array3 = abaloneNormalized.values
boxplot(array3)
show()
