#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    使用 sklearn 的随机森林算法
'''

import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn import ensemble
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

print "开始进行训练"
mseOos = []
nTreeList = range(50, 500, 10)
for iTree in nTreeList:
    depth = 2
    maxFeat = 3
    '''
    生成随机森林模型，比较重要的参数
    n_estimators {int} 指定集成方法中生成的决策树数量，默认为 10
    max_depth {int|None} 如果设置为 None 决策树会生长知道叶子节点为空或者节点所含数据小于 min_samples_split
    min_sample_split {int} 默认为 2，每个节点至少包含的数据实例数量
    max_features {int|float|str} 每次进行分割的时候，考虑多少个属性值
    random_state {int|RamdonState|None} 随机种子
    '''
    wineRFModel = ensemble.RandomForestRegressor(n_estimators=iTree,
        max_depth=depth, max_features=maxFeat, oob_score=False, random_state=531)
    wineRFModel.fit(xTrain, yTrain)
    prediction = wineRFModel.predict(xTest)						# 使用模型进行预测
    mseOos.append(mean_squared_error(yTest, prediction))		# 计算方差

print("MSE")
print(mseOos[-1])

''' 输出方差随决策树数量增加的变化趋势 '''
plot.plot(nTreeList, mseOos)
plot.xlabel(u"生成模型数量")
plot.ylabel(u"均方误差")
plot.show()

fetureImportance = wineRFModel.feature_importances_
fetureImportance = fetureImportance / fetureImportance.max()
sorted_idx = numpy.argsort(fetureImportance)
barPos = numpy.arange(sorted_idx.shape[0]) + .5
plot.barh(barPos, fetureImportance[sorted_idx], align='center')
plot.yticks(barPos, names[sorted_idx])
plot.xlabel(u'属性权重')
plot.show()
