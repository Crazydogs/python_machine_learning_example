#!/usr/bin/python
# -*- coding: UTF-8 -*-

'手工实现梯度提升'

import numpy
import matplotlib.pyplot as plot
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor  # 决策树回归
from math import floor
import random

# 生成数据集
nPoints = 1000
xPlot = [(float(i) / float(nPoints) - 0.5) for i in range(nPoints + 1)]
x = [[s] for s in xPlot]
numpy.random.seed(1)
y = [s + numpy.random.normal(scale=0.1) for s in xPlot]
# 展示数据
plot.figure()
plot.plot(x, y)
plot.xlabel('x')
plot.ylabel('y')
plot.show()

# 采样
nSample = int(nPoints * 0.30)   # 采样率为 30%
idxTest = random.sample(range(nPoints), nSample)
idxTest.sort()
idxTran = [idx for idx in range(nPoints) if not (idx in idxTest)]

# 测试集与训练集
xTrain = [x[r] for r in idxTran]
yTrain = [y[r] for r in idxTran]
xTest = [x[r] for r in idxTest]
yTest = [y[r] for r in idxTest]

# 参数设定
numTreeMax = 30; treeDepth = 5
# 模型数组
modelList = []
# 预测值
preList = []
# 步长控制
eps = 0.3
# 初始化残差
residuals = list(yTrain)

for iTree in range(numTreeMax):
    modelList.append(DecisionTreeRegressor(max_depth=treeDepth))
    # 使用训练集属性与残差对模型进行训练
    modelList[-1].fit(xTrain, residuals)

    # 使用本轮训练的模型对训练集进行预测
    latestInSamplePrediction = modelList[-1].predict(xTrain)
    # 更新残差，旧残差减去一定比例的本轮预测值
    residuals = [residuals[i] - eps * latestInSamplePrediction[i] \
        for i in range(len(residuals))]
    # 样本外预测值
    latestOutSamplePrediction = modelList[-1].predict(xTest)
    preList.append(latestOutSamplePrediction)

# 均方误差
mse = []
allPredictions = []
for iModels in range(len(modelList)):
    # 计算提升树模型的预测值须要将多轮迭代模型的结果进行求和
    pred = []
    for iPred in range(len(xTest)):
        # 这里求和的原理可以看一下等比数列求和
        # 不过仅当末项相对首项可忽略的情况下成立
        pred.append(sum(
            [preList[i][iPred] for i in range(iModels + 1)]
        ) * eps)

    allPredictions.append(pred)
    errors = [(yTest[i] - pred[i]) for i in range(len(yTest))]
    mse.append(sum([e*e for e in errors]) / len(yTest))

nModels = [i + 1 for i in range(len(modelList))]

plot.plot(nModels, mse)
plot.axis('tight')
plot.xlabel(u'迭代次数')
plot.ylabel(u'均方误差')
plot.ylim((0.0, max(mse)))
plot.show()

plotList = [0, 7, 14]
lineType = [':', '-.', '--']
plot.figure()
for i in range(len(plotList)):
    iPlot = plotList[i]
    textLegend = str(iPlot) + u'轮迭代的预测值'
    plot.plot(xTest, allPredictions[iPlot], label=textLegend, linestyle=lineType[i])
plot.plot(xTest, yTest, label=u'实际值', alpha=0.25)
plot.legend(bbox_to_anchor=(1,0.3))
plot.axis('tight')
plot.xlabel(u'x 值')
plot.xlabel(u'预测值')
plot.show()
