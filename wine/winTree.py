#!/usr/bin/python
# -*- coding: UTF-8 -*-

'使用二元决策树'

import numpy
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plot

# 读取数据
dataFile = open('wine.txt')
wine = pandas.read_csv(dataFile, header=0, sep=';')

nRows = len(wine.index)
nCols = len(wine.columns)

xList = wine.iloc[0: ,0:nCols-1];
labels = wine.iloc[0: ,nCols-1];

# 训练二元决策树
wineTree = DecisionTreeRegressor(max_depth=3)
wineTree.fit(xList, labels)

# 输出信息
print "参数权重: "
print wineTree.feature_importances_

print xList.iloc[0, 0:].values.reshape(1, -1)
print wineTree.predict(xList.iloc[0, 0:].values.reshape(1, -1))




# 生成数据观察二元决策树的预测

nPoints = 100   # 生成 100 组数据

xPlot = [(float(i) / float(nPoints) - 0.5) for i in range(nPoints + 1)]
x = [[s] for s in xPlot]

# y 等于 x 加上随机噪声
numpy.random.seed(1)
y = [s + numpy.random.normal(scale=0.1) for s in xPlot]

# 使用不同深度的二元决策树进行预测
# 二元决策在图中造成多个阶梯
for i in range(1, 4):
    simpoleTree = DecisionTreeRegressor(max_depth=i)
    simpoleTree.fit(x, y)
    yPre = simpoleTree.predict(x)

    plot.figure()
    plot.plot(xPlot, y, label='真实值')
    plot.plot(xPlot, yPre, label='预测值', linestyle='--')
    plot.axis('tight')
    plot.show()
