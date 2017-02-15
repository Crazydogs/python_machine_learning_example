#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 分类器性能分析

import pandas as pd
import numpy
import random
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl

def confusionMatrix(predicted, actual, threshold):
    """ 计算混淆矩阵

    predicted {arr} 预测值
    actual {arr} 实际值
    threshold {float} 判断阈值
    """
    if len(predicted) != len(actual) : return -1
    tp = 0.0    # 真正 预测与实际值均为正
    fp = 0.0    # 假正 预测为正，实际为负
    tn = 0.0    # 真负 预测与实际值均为负
    fn = 0.0    # 假负 预测为负，实际为正
    for i in range(len(actual)):
        # 标签值为 1.0，正例
        if actual[i] > 0.5:
            if predicted[i] > threshold:
                tp += 1.0       # 真正例
            else:
                fn += 1.0       # 假负例
        else:
            if predicted[i] < threshold:
                tn += 1.0       # 真负例
            else:
                fp += 1.0       # 假正例
    rtn = [tp, fn, fp, tn]
    return rtn

dataFile = open('data.txt')
rocksVMines = pd.read_csv(dataFile, header=None, prefix='V')
# 行数
nrows = len(rocksVMines.index)
indices = range(nrows)
# 列数
ncols = len(rocksVMines.columns)

# 将数据分离成属性和标签
xList = []
labels = []
for i in range(nrows):
    # 属性
    xRow = rocksVMines.iloc[i, 0:ncols - 1].values
    floatRow = [float(num) for num in xRow]
    xList.append(floatRow)
    # 标签
    labelRow = rocksVMines.iloc[i, ncols - 1]
    floatRow = [(1.0 if t == 'M' else 0.0) for t in labelRow]
    labels.append(floatRow)

# 测试集
xListTest = [xList[i] for i in indices if i%3 == 0]
labelsTest = [labels[i] for i in indices if i%3 == 0]
# 训练集
xListTrain = [xList[i] for i in indices if i%3 != 0]
labelsTrain = [labels[i] for i in indices if i%3 != 0]

xTrain = numpy.array(xListTrain); yTrain = numpy.array(labelsTrain)
xTest = numpy.array(xListTest); yTest = numpy.array(labelsTest)

# 线性模型 最小二乘法
rocksVminesModel = linear_model.LinearRegression()
# 训练线性模型
rocksVminesModel.fit(xTrain, yTrain)

# 样本内混淆矩阵
trainingPredictions = rocksVminesModel.predict(xTrain)
confusionMatTrain = confusionMatrix(trainingPredictions, yTrain, 0.5)
print('样本内混淆矩阵')
print("真正tp = " + str(confusionMatTrain[0]))
print("假负fn = " + str(confusionMatTrain[1]))
print("假正fp = " + str(confusionMatTrain[2]))
print("真负tn = " + str(confusionMatTrain[3]))

# 样本外数据混淆矩阵
testPredictions = rocksVminesModel.predict(xTest)
confusionMatTrain = confusionMatrix(testPredictions, yTest, 0.5)
print('样本外混淆矩阵')
print("真正tp = " + str(confusionMatTrain[0]))
print("假负fn = " + str(confusionMatTrain[1]))
print("假正fp = " + str(confusionMatTrain[2]))
print("真负tn = " + str(confusionMatTrain[3]))

# ROC 曲线
fpr, tpr, thresholds = roc_curve(yTrain, trainingPredictions)
roc_auc = auc(fpr, tpr)
print ('训练集的 AUC：%f' % roc_auc)
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('in sample ROC')
pl.legend(loc='lower right')
pl.show()

fpr, tpr, thresholds = roc_curve(yTest, testPredictions)
roc_auc = auc(fpr, tpr)
print ('测试集的 AUC：%f' % roc_auc)
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0,1], [0,1], 'k-')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('out of sample ROC')
pl.legend(loc='lower right')
pl.show()
