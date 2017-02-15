#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd                     # 一个用于数据处理的库
import numpy
import random
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl

# 计算混淆矩阵
def confusionMatrix(predicted, actual, threshold):
    if len(predicted) != len(actual):
        return -1
    tp = 0.0; fp = 0.0; tn = 0.0; fn = 0.0
    for i in range(len(actual)):
        if actual[i] > 0.5:
            if predicted[i] > threshold:
                tp += 1.0
            else:
                fn += 1.0
        else:
            if predicted[i] < threshold:
                tn += 1.0
            else:
                fp += 1.0
    rtn = [tp, fn, fp, tn]
    return rtn

# 读取 csv 格式的数据
dataFile = open('data.txt')
data = pd.read_csv(dataFile, header=None, prefix='V')


xList = []
labels = []

for line in dataFile:
    print(line)
    row = line.strip().split(",")
    # 将分类数据转化为数值数据
    if (row[-1] == 'M'):
        labels.append(1.0)
    else:
        labels.append(0.0)
    row.pop()
    # 批量将数据转型为浮点数
    floatRow = [float(num) for num in row]
    xList.append(floatRow)

indices = range(len(xList))
# 数据分成三份，两份作为训练数据，一份用于检验结果
xListTest = [xList[i] for i in indices if i%3 == 0]
xListTrain = [xList[i] for i in indices if i%3 != 0]

labelsTest = [labels[i] for i in indices if i%3 == 0]
labelsTrain = [labels[i] for i in indices if i%3 != 0]

xTrain = numpy.array(xListTrain)
yTrain = numpy.array(labelsTrain)
xTest = numpy.array(xListTest)
yTest = numpy.array(labelsTest)

print("Shape of xTrain array: ", xTrain.shape)
print("Shape of yTrain array: ", yTrain.shape)
print("Shape of xTest array: ", xTest.shape)
print("Shape of yTest array: ", yTest.shape)

print(xTrain)
