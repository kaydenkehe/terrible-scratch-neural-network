#This script creates training and testing data sets
import random

def doTheThing(trainingSize, pointLowerBound, pointUpperBound, m, b):
    xList = []
    yList = []
    for i in range(trainingSize):
        xList.append(random.randint(pointLowerBound * 100, pointUpperBound * 100) / 100)
        yList.append(random.randint(pointLowerBound * 100, pointUpperBound * 100) / 100)
    pointTuple = [(xList[i], yList[i]) for i in range(trainingSize)]
    dataSet = []
    for x,y in pointTuple:
        ans = 0
        if y < m * x + b: ans = -1
        if y >= m * x + b: ans = 1
        dataSet.append([x, y, ans])
    return dataSet