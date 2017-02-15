#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 红酒问题的折线图

import pandas
import pylab
import matplotlib.pyplot
from math import exp

dataFile = open('wine.txt')
# 指明文件第一行为列头
wine = pandas.read_csv(dataFile, header=0, sep=";")

summary = wine.describe()
print(summary)
# 行数
nrows = len(wine.index)
# 列数
tasteCol = len(summary.columns)
meanTaste = summary.iloc[1, tasteCol - 1]   # 平均得分
sdTaste = summary.iloc[2, tasteCol - 1]     # 得分方差

for i in range(tasteCol):
    # 对每一列进行归一操作
    mean =  summary.iloc[1, i]
    sd =  summary.iloc[2, i]
    wine.iloc[:,i:(i+1)] = (wine.iloc[:,i:(i+1)] - mean) / sd

for i in range(nrows):
    dataRow = wine.iloc[i, 0:(tasteCol - 1)]
    labelColor = 1.0/(1.0 + exp(-wine.iloc[i, (tasteCol - 1)]))
    # 绘制平行折线图
    dataRow.plot(color=matplotlib.pyplot.cm.RdYlBu(labelColor), alpha=0.5)

matplotlib.pyplot.show()
