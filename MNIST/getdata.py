
# PyTorch is built on python and torch library
# PyTorch is an advanced framework of DeepLearning
import torch
# 基于numpy的工具，解决数据分析任务而创建，纳入了大量的维度数组与矩阵计算 即：大量数学函数库
import pandas as pd
# Machine Learning library
from sklearn.model_selection import train_test_split


def read_data():
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    return train_data, test_data

# print(train_set.head())

def split_data():
    # train_test_split 是Scikit Learn中的方法
    # test_size: 可以为浮点、整数或None(default: None) 若为浮点数：表示训练集占总样本的百分比；整数：表示训练样本的样本数；None：train_size自动被设置为75
    # data.iloc[0] means 取第一行数据 data.ilon[:, 1]means 取第一列所有行 data.ilon[:, 1:]means 取第一到最后列（除去第零列）所有行
    # 把数据分为训练集和测试集 训练集 训练集标签 测试集 测试集标签
    train_set, _ = read_data()
    train_images, val_images, train_labels, val_labels = train_test_split(train_set.iloc[:, 1:], train_set.iloc[:, 0], test_size=0.2)

    # Now: train_images(33600, 784) val_images(8400, 784) train_labels(33600,) val_labels(8400,) 784 = 28 * 28
    # Reset indices so the Dataset can find them with __getitem__ easily
    # Pandas 中的重置索引方法  drop: 不将索引插入数据框列 inplace: 是否修改dataframe或者创建一个新的
    train_images.reset_index(drop=True, inplace = True)
    val_images.reset_index(drop=True, inplace = True)
    train_labels.reset_index(drop=True, inplace = True)
    val_labels.reset_index(drop=True, inplace = True)
    return train_images, val_images, train_labels, val_labels