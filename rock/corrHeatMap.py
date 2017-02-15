#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 绘制属性相关性热图

import pandas as pd                     # 一个用于数据处理的库
from pandas import DataFrame            # pandas 中的一种数据结构，表达一个表格结构
import matplotlib.pyplot as plot        # 绘图

# 读取 csv 格式的数据
dataFile = open('data.txt')
rocksVMines = pd.read_csv(dataFile, header=None, prefix='V')

# 获取数据的相关性矩阵
corMat = DataFrame(rocksVMines.corr())

plot.pcolor(corMat)
plot.show()
