#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 绘制属性对交会图

import pandas as pd                     # 一个用于数据处理的库
import matplotlib.pyplot as plot        # 绘图
from math import sqrt

# 读取 csv 格式的数据
dataFile = open('data.txt')
rocksVMines = pd.read_csv(dataFile, header=None, prefix='V')

dataRow2 = rocksVMines.iloc[1, 0:60]
dataRow3 = rocksVMines.iloc[2, 0:60]

# 绘制交会图
# plot.scatter(dataRow2, dataRow3)
# plot.show()

dataRow21 = rocksVMines.iloc[20, 0:60]

# 绘制交会图
# plot.scatter(dataRow2, dataRow21)
# plot.show()

# 计算平均值
mean2 = 0.0; mean3 = 0.0; mean21 = 0.0
numElt = len(dataRow2)
for i in range(numElt):
    mean2 += dataRow2[i] / numElt
    mean3 += dataRow3[i] / numElt
    mean21 += dataRow21[i] / numElt

# 计算方差
var2 = 0.0; var3 = 0.0; var21 = 0.0
for i in range(numElt):
    var2 += (dataRow2[i] - mean2) * (dataRow2[i] - mean2) / numElt
    var3 += (dataRow3[i] - mean3) * (dataRow3[i] - mean3) / numElt
    var21 += (dataRow21[i] - mean21) * (dataRow21[i] - mean21) / numElt

# 计算皮尔逊相关系数
corr23 = 0.0; corr221 = 0.0
for i in range(numElt):
    corr23 += (dataRow2[i] - mean2) * (dataRow3[i] - mean3) / (sqrt(var2 * var3) * numElt)
    corr221 += (dataRow2[i] - mean2) * (dataRow21[i] - mean21) / (sqrt(var2 * var21) * numElt)

print(corr23)
print(corr221)
