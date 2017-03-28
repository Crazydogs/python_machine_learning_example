#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    使用 sklearn 的梯度提升算法
'''

import pandas
import numpy
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pylab as plot

# 读取数据
dataFile = open('wine.txt')
wine = pandas.read_csv(dataFile, header=0, sep=';')

nRows = len(wine.index)
nCols = len(wine.columns)

xList = wine.iloc[0:, 0:nCols - 1]
labels = wine.iloc[0:, nCols - 1]

names = numpy.array(wine.columns)
X = numpy.array(xList)
Y = numpy.array(labels)

# 通过 train_test_split 来分隔测试集与训练集
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.30, random_state=531)

nEst = 2000
depth = 7
learnRate = 0.01
subSamp = 0.5
wineGBMModel = ensemble.GradientBoostingRegressor(
    n_estimators=nEst, max_depth=depth, learning_rate=learnRate, subsample=subSamp, loss='ls')
wineGBMModel.fit(xTrain, yTrain)

msError = []
predictions = wineGBMModel.staged_predict(xTest)
for p in predictions:
    msError.append(mean_squared_error(yTest, p))

plot.figure()
plot.plot(range(1, nEst + 1), wineGBMModel.train_score_, label=u"训练集均方误差")
plot.plot(range(1, nEst + 1), msError, label=u"测试集均方差")
plot.legend(loc="upper right")
plot.xlabel(u'决策树数量')
plot.ylabel(u'均方差(MSE)')
plot.show()