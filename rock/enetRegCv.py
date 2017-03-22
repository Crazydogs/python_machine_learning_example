#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 使用 ElasticNet 回归构建二分类器

import pandas as pd
from math import sqrt, fabs, exp
import matplotlib.pyplot as plot

from sklearn.linear_model import enet_path, ElasticNetCV
from sklearn import preprocessing
# 用于判断分类器性能的指标
from sklearn.metrics import roc_auc_score, roc_curve
import numpy

dataFile = open('data.txt')
rocksVMines = pd.read_csv(dataFile, header=None, prefix='V')

nRows = len(rocksVMines.index)      # 行数
nCols = len(rocksVMines.columns)    # 列数

xList = [];     # 属性集
labels = [];    # 标签集
for i in range(nRows):
    xRow = rocksVMines.iloc[i, 0:nCols - 1].values
    xList.append([float(num) for num in xRow])
    label = rocksVMines.iloc[i, nCols - 1]
    # 将二分类标签转化为数值
    if label == 'M':
        labels.append(1.0)
    elif label == 'R':
        labels.append(0.0)

# 归一化数据 归一化公式为 (val - mean(x)) / std(x)
xList = preprocessing.scale(xList)
labels = preprocessing.scale(labels)

X = numpy.array(xList)
Y = numpy.array(labels)

# 进行 10 折交叉校验
rocksVminesModel = ElasticNetCV(cv=10,l1_ratio=0.8,fit_intercept=False).fit(X, Y)
# 绘制平均方差随 alphas 变化的过程
plot.plot(rocksVminesModel.alphas_, rocksVminesModel.mse_path_, ':')
plot.plot(rocksVminesModel.alphas_, rocksVminesModel.mse_path_.mean(axis=-1),
    label=u'平均的跨折方差')
plot.axvline(rocksVminesModel.alpha_, linestyle='--', label=u'最佳 Aplpha')
plot.semilogx()      # x 坐标用对数级别展示
plot.xlabel('alpha')
plot.ylabel(u'均方差')
plot.show()

# 使用 elasticNet 对数据进行回归
alphas, coefs, _ = enet_path(X, Y, l1_ratio=0.8, fit_intercept=False, return_models=False)
alphas = numpy.array(alphas)
coefs = numpy.array(coefs)

# 绘制系数随 alphas 变化的过程
for coefs_i in coefs:
    plot.plot(alphas, coefs_i)
plot.axvline(rocksVminesModel.alpha_, linestyle='--', label=u'最佳Aplpha')
plot.legend()
ax = plot.gca()
ax.invert_xaxis()       # 翻转 x 轴
plot.semilogx()         # x 坐标用对数级别展示
plot.show()
