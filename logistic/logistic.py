import numpy as np
import matplotlib.pyplot as plt




#def loadDataSet():
dataMat = []                                                        #创建数据列表
labelMat = []                                                        #创建标签列表
fr = open('data.txt')                                            #打开文件
npfr = np.loadtxt('data.txt')
#print(npfr.shape)
#print(npfr)
for line in fr.readlines():                                            #逐行读取
    lineArr = line.strip().split()                                    #去回车，放入列表
#    print(type(lineArr))
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])        #添加数据
#    print(lineArr)
    labelMat.append(int(lineArr[21]))                                #添加标签
#    print(labelMat)
fr.close()                                                            #关闭文件
print(dataMat, labelMat)