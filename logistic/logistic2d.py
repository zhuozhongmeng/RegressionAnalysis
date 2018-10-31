import numpy as np
import matplotlib.pyplot as plt
import random


# 定义函数用来做样本的拆分
def m_getdata(data):
    m, n = data.shape
    data_x = data[:, :n - 1]
    data_y = data[:, -1]
    data_x_0_view = []
    data_x_1_view = []

    for i in range(len(data_y)):
        if data_y[i] == 0:
            data_x_0_view.append(data_x[i])
        elif data_y[i] == 1:
            data_x_1_view.append(data_x[i])
        else:
            print("样本中，第", i + 1, "行数据的分类标签并非0/1，请检查数据")
            break
    return data_x, data_y, m, n, data_x_0_view, data_x_1_view


# sigmoid函数
def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))


# 回归函数，type=0时使用批量逻辑回归，type=1时使用随机逻辑回归,只输出w矩阵，
def logistic(data_x, data_y, type=0):
    mm, mn = data_x.shape  # 这里需要重新读取输入矩阵的形状数据m,n，函数使用参数要独立
    w = np.ones(mn)
    delta_sum_view = []
    w_view = np.ones([times, mn])
    w_view = np.array(w_view)
    w_view = w_view.reshape([times, mn])

    for t in range(times):

        for i in range(mn):
            w_view[t, i] = w[i]
        if type == 0:  # 使用批量逻辑回归
            delta_sum_view.append(np.dot((sigmoid(np.dot(data_x, w)) - data_y), data_x))
            w -= rate * np.dot((sigmoid(np.dot(data_x, w)) - data_y), data_x) / mm

        if type == 1:  # 使用随机逻辑回归
            datamap = list(range(mm))
            delta_sum = 0
            for i in range(mm):
                randIndex = int(random.uniform(0, len(datamap)))
                # print(len(datamap))
                h = sigmoid(sum(data_x[randIndex] * w))
                delta = data_y[randIndex] - h
                delta_sum += delta ** 2
                # d = np.sum(data_x, 0)
                w += rate * delta * data_x[randIndex]
                del (datamap[randIndex])
            delta_sum_view.append(delta_sum)

        w2 = []
        if t % 100 == 0:
            for a in range(-3, 3, ):
                w2.append((-w[0] - (w[1] * a)) / w[2])
            plt.plot(data_x_0_view[:, 0], data_x_0_view[:, 1], 'x')
            plt.plot(data_x_1_view[:, 0], data_x_1_view[:, 1], 'x')
            plt.plot(range(-3, 3), w2)
            plt.pause(0.0001)
            plt.close()
            print(t / times * 100, "%")
            # print(delta ** 2)
    return w, delta_sum_view, w_view


def m_plt_show(w, data, delta_sum_view, w_view):
    data_x, data_y, m, n, data_x_0_view, data_x_1_view = m_getdata(data)
    data_x_0_view = np.array(data_x_0_view)
    data_x_1_view = np.array(data_x_1_view)

    w2 = []
    print(w_view)
    for a in range(-3, 3, ):
        w2.append((-w[0] - (w[1] * a)) / w[2])
    plt.subplot(221)
    plt.plot(range(-3, 3), w2)
    plt.plot(data_x_0_view[:, 0], data_x_0_view[:, 1], 'x')
    plt.plot(data_x_1_view[:, 0], data_x_1_view[:, 1], 'x')
    plt.subplot(222)
    plt.plot(range(times), delta_sum_view)
    plt.subplot(223)
    plt.plot(range(times), w_view[:, 1])
    plt.subplot(224)
    plt.plot(range(times), w_view)
    plt.show()


rate = 0.01
times = 10000

data = np.loadtxt("data2d.txt")
data_x, data_y, m, n, data_x_0_view, data_x_1_view = m_getdata(data)
data_x = np.insert(data_x, 0, values=1, axis=1)

data_x_0_view = np.array(data_x_0_view)
data_x_1_view = np.array(data_x_1_view)

w, delta_sum_view_m, w_view = logistic(data_x, data_y, type=0)

show = m_plt_show(w, data, delta_sum_view_m, w_view)
