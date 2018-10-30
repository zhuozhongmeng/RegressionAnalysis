import numpy as np
import matplotlib.pyplot as plt
import random

data = np.loadtxt("data2d.txt")
m, n = data.shape
print(data.shape)
data_x = data[:, 0:n - 1]
data_y = data[:, -1]
data_x = np.insert(data_x, 0, values=1, axis=1)
view_0 = []
view_1 = []
for i in range(m):
    if data_y[i] == 0:
        view_0.append(list(data_x[i, :]))
    if data_y[i] == 1:
        view_1.append(list(data_x[i, :]))
view_0 = np.array(view_0)
view_1 = np.array(view_1)
w = np.ones(n)
rate = 0.01
times = 10000


def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))


delta_sum_view = []
w_view = np.ones([times, 3])

for t in range(times):
    w2 = []
    for i in range(3):
        w_view[t, i] = w[i]

    # print(data_x[i,])
    datamap = list(range(m))
    delta_sum_view.append(np.sum(sigmoid(np.dot(data_x, w)) - data_y))
    w -= rate * np.dot((sigmoid(np.dot(data_x, w)) - data_y), data_x) / m
    # for i in range(m):
    #    randIndex = int(random.uniform(0, len(datamap)))
    #    #print(len(datamap))
    #    h = sigmoid(sum(data_x[randIndex] * w))
    #    delta = data_y[randIndex] - h
    # d = np.sum(data_x, 0)
    #    w += rate * delta * data_x[randIndex]
    #    del(datamap[randIndex])
    if t % 10 == 0:
        for a in range(-3, 3, ):
            w2.append((-w[0] - (w[1] * a)) / w[2])
        plt.plot(view_0[:, 1], view_0[:, 2], 'x')
        plt.plot(view_1[:, 1], view_1[:, 2], 'x')
        plt.plot(range(-3, 3), w2)
        plt.pause(0.0001)
        plt.close()
        print(t / times * 100, "%")
        #print(delta)
    #delta_sum_view.append(delta)

w_view = np.array(w_view)
w_view = w_view.reshape([times, 3])
w2 = []
print(w_view)
for a in range(-3, 3, ):
    w2.append((-w[0] - (w[1] * a)) / w[2])
plt.subplot(221)
plt.plot(range(-3, 3), w2)
plt.plot(view_0[:, 1], view_0[:, 2], 'x')
plt.plot(view_1[:, 1], view_1[:, 2], 'x')
plt.subplot(222)
plt.plot(range(times), delta_sum_view)
plt.subplot(223)
plt.plot(range(times), w_view[:, 1])
plt.subplot(224)
plt.plot(range(times), w_view)
plt.show()
