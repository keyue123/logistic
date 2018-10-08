#!/usr/bin/python                                                                                                                                                                                                                    
#coding=utf-8

#File Name: test.py
#Author   : john
#Mail     : john.y.ke@mail.foxconn.com 
#Created Time: Sat 01 Sep 2018 05:38:56 PM CST

import matplotlib.pyplot as plt
import pandas as pd
from math import *
from numpy import *

def drawPoint(fileName):
    datas = pd.read_table(fileName, header=None, encoding='utf-8', names=['X', 'Y', 'Result'])
    for data in datas.values:
        if data[2] == 0:
            plt.scatter(data[0], data[1], c='b')
        else:
            plt.scatter(data[0], data[1], c='r')

    plt.show()

def loadDataSet0(fileName):
    dataMat = []    #原始数据
    labelMat = []   #数据标签
    fr = pd.read_table(fileName, header=None, encoding='utf-8') #打开数据集
    for line in fr.values:
        dataMat.append([1.0, float(line[0]), float(line[1])])  # 我们将X0的值设为1.0
        labelMat.append(int(line[2]))  # 结果

    return dataMat, labelMat

def sigmoid(inX):
    #return 2 * 1.0/(1+exp(-2*inX)) - 1
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #转换为矩阵
    labelMat = mat(classLabels).transpose() #首先将数组转换为矩阵，然后再将行向量转置为列向量

    m, n = shape(dataMatrix)    # m数据量  n特征数
    alpha = 0.001   #向目标移动的步长
    maxCycles = 500 # 迭代次数
    weights = ones((n, 1))   #生成一个全为1的矩阵
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)     #矩阵乘法
        error = (labelMat - h)              #向量相减
        weights = weights + alpha * dataMatrix.transpose() * error #矩阵乘法，最后得到回归系数

    return array(weights)

# 随机梯度上升算法
def stocGradAscent(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)

    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.0001    # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))

            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]]*weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            #del(dataIndex[randIndex])

    return weights

def plotBestFit(dataArr, labelMat, weights):
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, c='red', marker='s') #结果为1，红色
    ax.scatter(xcord2, ycord2, c='green')           #结果为0，绿色
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]

    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

#分类
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def loadDataSet(fileName):
    frTrain = open(fileName)
    trainingSet = []
    trainingLabels = []

    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    return trainingSet, trainingLabels

def testResult(fileName):
    errorCount = 0
    numTestVec = 0.0

    frTest = open(fileName)
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print ("the error rate of this test is: %f" % errorRate)

    return errorRate

if __name__ == '__main__':
    '''
    dataMat, labelMat = loadDataSet0("C:\\Users\\John\\Desktop\\DataBooks\\MachineLearning\\Ch05\\TestSet.txt") #收集并准备数据

    dataArr = array(dataMat)
    weights = gradAscent(dataArr, labelMat)#训练模型

    plotBestFit(dataArr, labelMat, weights) #数据可视化
    '''
    trainFilePath = 'C:\\Users\\John\\Desktop\\DataBooks\\MachineLearning\\Ch05\\horseColicTraining.txt'
    testFilePath = 'C:\\Users\\John\\Desktop\\DataBooks\\MachineLearning\\Ch05\\horseColicTest.txt'
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        trainingSet, trainingLabels = loadDataSet(trainFilePath)
        trainWeights = stocGradAscent(array(trainingSet), trainingLabels, 500)  # 使用随机梯度下降算法 求得在此数据集上的最佳回归系数
        errorSum += testResult(testFilePath)

    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))