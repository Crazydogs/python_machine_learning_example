#!/usr/bin/python
# -*- coding: UTF-8 -*-

'使用 sklearn 中的套索模型'

import pandas
import numpy
from sklearn import datasets, linear_model, preprocessing
from sklearn.linear_model import LassoCV        # 套索回归模型
from math import sqrt
import matplotlib.pyplot as plt

# 读取数据
dataFile = open('wine.txt')
wine = pandas.read_csv(dataFile, header=0, sep=';')
print(wine.describe())

# 归一化数据 归一化公式为 (val - mean(x)) / std(x)
scaledWine = preprocessing.scale(wine)
df = pandas.DataFrame(scaledWine)
print(df.describe())

names = wine.columns.values

rows = len(wine.index)
cols = len(wine.columns)

xList = []
labels = []
for i in range(rows):
    # 属性
    row = scaledWine[i][0 : cols - 1]
    xList.append(row)
    # 预测值
    label = float(scaledWine[i][cols - 1])
    labels.append(label)


# 使用 10 折交叉验证
X = numpy.array(xList)
Y = numpy.array(labels)
wineModel = LassoCV(cv=10).fit(X, Y)    # cv 为折数

plt.figure()
plt.plot(wineModel.alphas_, wineModel.mse_path_, ':')
plt.plot(wineModel.alphas_, wineModel.mse_path_.mean(axis=-1),
    label=u'平均的跨折方差')
plt.axvline(wineModel.alpha_, linestyle='--', label=u'最佳 Aplpha')
plt.semilogx()      # x 坐标用对数级别展示
plt.legend()        # 显示 label 标签
ax = plt.gca()
ax.invert_xaxis()   # 翻转 x 轴
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel(u'均方差')
plt.show()

print u'使平均均方差最小的 alpha 为', wineModel.alpha_
print u'最小均方差为', min(wineModel.mse_path_.mean(axis=-1))

# 在全数据集进行训练
alphas, coefs, _ = linear_model.lasso_path(X, Y, return_models=False)

# 绘制系数随 alpha 值变化的趋势
plt.plot(alphas, coefs.T)
plt.xlabel('Alphas')
plt.ylabel('Coefficients')
plt.axis('tight')
plt.semilogx()      # x 坐标用对数显示
ax = plt.gca()
ax.invert_xaxis()   # 翻转 x 轴
plt.show()
