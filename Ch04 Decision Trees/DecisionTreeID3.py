from math import log
import operator
import pandas as pd
import numpy as np

def createDataSet():
    """
    简单鱼的鉴定数据集
    :return: 数据集，特征种类
    """
    #特征：1.不浮出水面是否可以生存 No surfacing 2.是否有脚蹼 Flippers 3.属于鱼类 Fish

    readData = pd.read_csv('./watermelonData.csv', encoding='gbk')
    ndarryData = np.array(readData)
    dataSet = ndarryData.tolist()
    print(dataSet)
    labels = []
    for item in readData: labels.append(item)
    print(labels)

    return dataSet, labels

'''
def createDataSet():
    """
    简单鱼的鉴定数据集
    :return: 数据集，特征种类
    """
    #特征：1.不浮出水面是否可以生存 No surfacing 2.是否有脚蹼 Flippers 3.属于鱼类 Fish
    dataSet = [[1, 1, 'yes'],
           [1, 1, 'yes'],
           [1, 0, 'no'],
           [0, 1, 'no'],
           [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels
'''

def calcShannonEnt(dataSet):
    """
    计算数据集的香农熵
    :param dataSet: 数据集
    :return: 香农熵
    """
    #获取数据集样本总数
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        #取此样本的最后一个特征值
        currentLabel = featVec[-1]
        #如果这个特征值之前没有出现，则添加到labelCounts中，且出现次数初始值为0
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        #这个特征值出现次数+1
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        #此特征值出现的次数/总次数 = 此特征值在样本总数中的概率
        prob = float(labelCounts[key])/numEntries
        #计算香农熵
        shannonEnt -= prob * log(prob, 2) #log base 2
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """
    按给定特征划分数据集
    :param dataSet: 数据集
    :param axis: 以axis列为标准划分
    :param value: 获取axis列的值为value的数据
    :return: 获取的数据
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            #.extend() 在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
            reducedFeatVec.extend(featVec[axis + 1:])
            #.append() 在列表末尾添加新的对象
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet: 数据集
    :return: 返回最好的信息增益
    """
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]  #create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    """
    排序字典
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    创建树
    :param dataSet: 数据集
    :param labels: 特征种类
    :return: 树
    """
    classList = [example[-1] for example in dataSet]
    #类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    #遍历完所有特征时返回出现次数最多的
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree




'''画图'''
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

#获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    #firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  #test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs += 1
    return numLeafs

#获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    #firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  #test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else: thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    #firstStr = myTree.keys()[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #坐标轴
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()



'''___main___'''

#获取初始数据
myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
print(myTree)

createPlot(myTree)