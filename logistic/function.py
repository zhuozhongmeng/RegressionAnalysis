import numpy as np
import time
import random
import matplotlib.pyplot as plt


# 默认返回样本不分类 #默认不增加偏置项数列 默认偏置为1，
def m_getdata(data, gettype=False, getinsert=False, bias=1):  # 默认返回样本不分类 #默认不增加偏置项数列 默认偏置为1
    m, n = data.shape
    data = np.array(data)
    data_x = data[:, 0:n - 1]
    data_y = data[:, -1]
    data_x_with_type_0 = []
    data_x_with_type_1 = []
    data_y_with_type_0 = []
    data_y_with_type_1 = []
    for i in range(m):
        if data_y[i] == 0:
            data_x_with_type_0.append(data_x[i])
            data_y_with_type_0.append(data_y[i])
        if data_y[i] == 1:
            data_x_with_type_1.append(data_x[i])
            data_y_with_type_1.append(data_y[i])
    if getinsert == True:
        data_x = np.insert(data_x, 0, values=bias, axis=1)
        data_x_with_type_0 = np.insert(data_x_with_type_0, 0, values=bias, axis=1)
        data_x_with_type_1 = np.insert(data_x_with_type_1, 0, values=bias, axis=1)

    if gettype == False:
        return data_x, data_y
    if gettype == True:  # 调用分类时，返回两个分类的对应数据
        return data_x_with_type_0, data_y_with_type_0, data_x_with_type_1, data_y_with_type_1


def m_sigmoid(inx):
    return 1 / (1 + np.exp(-inx))


# 逻辑回归函数，返回W
def m_logistic(data_x, data_y, type=0, rate=0.001, times=1000):
    time_start = time.clock()
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    m, n = data_x.shape
    lenm = len(data_x)
    lenp = len(data_y)
    if lenm != lenp:
        print("输入数据样本和结果样本行数不一致，请查验数据")
        print(lenm, lenp)
        return
    else:
        print("数据正常")

    print("继续代码")
    if type == 0:
        w = np.ones(n)
        delta_sum = []
        for t in range(times):
            w -= rate * np.dot((m_sigmoid(np.dot(data_x, w)) - data_y), data_x) / m

            x_temp = m_sigmoid(np.dot(data_x, w)) - data_y
            y_temp = abs(x_temp)
            delta_sum.append(np.sum(abs(m_sigmoid(np.dot(data_x, w)) - data_y)))
            # print(delta_sum)
        time_stop = time.clock()
        print("本次训练使用时间为", time_stop - time_start, "秒")
        plt.plot(range(times), delta_sum)
        plt.pause(1)
        plt.close()
        #print(w)

    if type == 1:
        w = np.ones(n)
        delta_sum = []
        for t in range(times):
            if t % 100 == 0:
                print(t * 100 / times, "%")
            delta_sum_square = 0
            simple_list = list(range(m))
            for i in range(len(simple_list)):
                randomMpa = int(random.uniform(0, len(simple_list)))
                # print(np.dot(data_x[randomMpa],w))
                w -= (m_sigmoid(np.dot(data_x[randomMpa], w)) - data_y[randomMpa]) * data_x[randomMpa] * rate / m
                delta_sum_square += (m_sigmoid(np.dot(data_x[randomMpa], w)) - data_y[randomMpa]) ** 2
                del simple_list[randomMpa]
            # print(delta_sum_square)
            delta_sum.append(delta_sum_square)
        time_stop = time.clock()
        print("本次训练使用时间为", time_stop - time_start, "秒")
        plt.plot(range(times), delta_sum)
        plt.pause(1)
        plt.close()

    return w


# 验证函数
def m_test_w(w, data_x, data_y):
    m, n = data_x.shape
    q = len(data_y)
    p = len(w)
    if q != m:
        print("数据格式有误，请查阅数据")

    if p != n:
        print("验证数据格式与训练格式不一致")

    y = m_sigmoid(np.dot(data_x, w))
    error = 0
    for i in range(m):
        if abs(y[i] - data_y[i]) >= 0.5:
            error += 1

    print("准确率为", (1 - error / m) * 100, "%")
    print(w)
