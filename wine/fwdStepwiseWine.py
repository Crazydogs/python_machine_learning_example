#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 逐步向前回归

import pandas
import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt

def xattrSelect(x, idxSet):
    """ 获取特征矩阵中的指定列
        x {arr} 特征矩阵，二维数组，每行代表一个样本
        idxSet {arr} 想提取的列 id
    """
    xout = []
    for row in x:
        xout.append([row[i] for i in idxSet])
        
dataFile = open('wine.txt')
wine = pandas.read_csv(dataFile, header=0, sep=';')
# 行数
nrows = len(wine.index)
indices = range(nrows)
# 列数
ncols = len(wine.columns)

# 将数据分离成属性和标签
xList = []
labels = []
names = wine.columns.values
for i in range(nrows):
    # 属性
    xRow = wine.iloc[i, 0:ncols - 1].values
    floatRow = [float(num) for num in xRow]
    xList.append(floatRow)
    # 标签
    labelRow = float(wine.iloc[i, ncols - 1])
    labels.append(labelRow)

# 测试集
xListTest = [xList[i] for i in indices if i%3 == 0]
labelsTest = [labels[i] for i in indices if i%3 == 0]
# 训练集
xListTrain = [xList[i] for i in indices if i%3 != 0]
labelsTrain = [labels[i] for i in indices if i%3 != 0]

# 特征列表
attributeList = []
index = range(len(xList[1]))
indexSet = set(index)
indexSeq = []
oosError = []

for i in index:
    attSet = set(attributeList)
    attTrySet = indexSet - attSet
    attTry = [ii for ii in attTrySet]
    errorList = []
    attTemp = []
    for iTry in attTry:
