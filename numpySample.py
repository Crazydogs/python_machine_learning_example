#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''numpy 基本功能
Pandas 基本功能 '''

import numpy as np

# 可以使用字符串生成矩阵
a = np.mat('1 2; 3 4')
print(a)
print('转置矩阵')
print(a.T)      # 转置矩阵
print('共轭转置')
print(a.H)      # 共轭转置
print('逆矩阵')
print(a.I)      # 逆矩阵
print('返回自身数据的2维数组的一个视图')
print(a.A)      # 返回自身数据的2维数组的一个视图

# 可以使用二维数组生成矩阵
b = np.mat([[5, 5], [6, 6]])
print(b)


c = np.bmat([a, b])
print(c)

print(np.random.rand(5,5))

print '=========='

from pandas import Series, DataFrame

# Series 是一组 Numpy 数据类型的数据还有与其相关的标签(索引)
print 'Series 基础用法'
e = Series(np.random.rand(3), index=['index1', 'index2', 'index3'])
print(e)

# 使用索引
print 'index2' in e
print e['index3']
print e.index
# 与字典类型相互转换
print '字典'
print e.to_dict()
print Series({'a': 2311, 'b': 31890, 'c': 282})

# DataFrame是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型(数值、字符串、布尔值等)
print 'DataFrame 基本用法'
f = DataFrame({
    'one': Series([1., 2., 3.], index=['a', 'b', 'c']),
    'two': Series([1., 2., 3.], index=['a', 'b', 'c'])
})
print f

# 各列的数据类型可以不同
g = np.zeros((2,), dtype=[('A', 'i4'), ('B', 'f4'), ('C', 'a10')])
g[:] = [(1,2,'Hello'), (2,3.,'world!')]     # 第一行中的 2 自动转成了浮点型
g = DataFrame(g)
# 可以按列进行操作
print '添加与删除列'
g['D'] = g['A'] / g['B']
print DataFrame(g)
del g['D']
print DataFrame(g)


