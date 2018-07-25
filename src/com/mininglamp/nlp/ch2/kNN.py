
from numpy import *
import numpy as np
import operator


def createDataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    '''
    简单进行kNN分类, 用欧氏距离计算距离
    :param inX: 输入数据的向量
    :param dataSet: 已有数据集
    :param labels: 已有数据集对应的标签
    :param k: 取 topK
    :return:
    '''
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet   # tile函数将inX进行重复, 横向dataSetSize次重复, 纵向不重复(与dataSet作减法)
    sqDiffMat = diffMat**2   # 平方处理
    sqDistances = sqDiffMat.sum(axis=1)  # axis=1 是每一行求和. axis=0是每一列求和.
    distances = sqDistances ** 0.5   # 得到输入数据与dataSet中每一项的距离
    sortedDistIndicies = np.argsort(distances)  # 对距离进行排序
    classCount = {}  # 创建一个map,记录topK的标签和count
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 排序类别字典
    return sortedClassCount[0][0]


# 约会网站配对
def file2matrix(filename):
    '''
    将文件转换为矩阵
    :param filename:
    :return:
    '''
    labelMap = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    with open(filename, 'r', encoding='utf-8') as f:
        arrayOLines = f.readlines()
        numberOfLines = len(arrayOLines)
        returnMat = zeros((numberOfLines, 3))  # 创建返回的矩阵 (n x 3的矩阵)
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0:3]
            index += 1
            classLabelVector.append(labelMap.get(listFromLine[-1]))   # 标签数组
        return returnMat, classLabelVector  # 返回矩阵和标签


import matplotlib
import matplotlib.pyplot as plt


def showDatingData(filename):
    '''
    用Matplotlib绘图
    :param filename:
    :return:
    '''
    datingDataMat,datingLabels = file2matrix(filename)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()


def autoNorm(dataSet):
    '''
    归一化处理, 飞行里程数和玩游戏的时间,在数值上差距很大,这样就非常影响最终距离的计算. 需要进行归一化
    最常用的归一化方法: newValue = (oldValue - min)/ (max-min)
    :param dataSet:
    :return:
    '''
    minVals = dataSet.min(0)    # 按列取最小值, 返回每列最小值组成的向量.
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]     # 行数
    normDataSet = dataSet - tile(minVals, (m, 1))   # 重复minVals, 行重复m次, 列重复1次.
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10   # 测试数据的比例
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')   # 加载数据源
    normMat, ranges, minVals = autoNorm(datingDataMat)   # 归一化
    m = normMat.shape[0]   # 获取行数
    numTestVecs = int(m * hoRatio)   # 获取测试的行数
    errorCount = 0.0   # 错误率
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :],
                                     normMat[numTestVecs:m, :],   # 测试数据之外的 90% 作为模型的数据(训练数据)
                                     datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is : %f" %(errorCount/float(numTestVecs)))



def classifyPerson():
    '''
    约会分类器的可用系统, 用户输入数据, 完成预测
    :return:
    '''
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLables = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLables, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


'''
以下算法用KNN进行图像识别, 识别0-9数字
'''


def img2vector(filename):
    '''
    将每一个图像(32*32)转换成1*1024的向量
    :param filename: 输入已转化数字文件的路径
    :return: 1 x 1024的向量
    '''
    returnVect = zeros((1,1024))
    with open(filename, 'r', encoding="UTF-8") as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i + j] = int(lineStr[j])
    return returnVect


if __name__ == '__main__':
    # dataSet, labels = createDataset()
    # print(classify0([0, 0], dataSet, labels, 3))
    datingFile = 'datingTestSet.txt'
    # datingDataMat, datingLabels = file2matrix(datingFile)
    # showDatingData(datingFile)

    # 测试模型
    # datingClassTest()

    # 实际应用
    classifyPerson()


    # print('123')

