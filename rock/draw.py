#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 绘制数据图像

import numpy as np              # 数学相关
import scipy.stats as stats     # scipy 是一个开源科学库
# import matplotlib.pyplot as plt
import pylab                    # 绘图使用

dataFile = open('data.txt')

xList = []
label = []

line = dataFile.readline()
while line:
    # 读取文件
    row = line.strip().split(',')
    xList.append(row)
    line = dataFile.readline()

colData = []
for row in xList:
    # 抽取第四列数据
    colData.append(float(row[3]))

colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)

stats.probplot(colData, dist = 'norm', plot=pylab)
pylab.show()
