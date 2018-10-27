import numpy as np
import matplotlib.pyplot as plt
import math

rate = 0.00000005

data_np = np.loadtxt('data.txt')  # 用np来读区数据
q, t = data_np.shape
data_label = data_np[0:q, -1]  # 样本输出，取数据最后一列
data_input = data_np[0:q, 0:t - 1, ]  # 样本输入，去掉样本最后一列
one_insert = np.ones(q)
print(one_insert)
data_input = np.insert(data_input, 0, values=one_insert,axis=1)
print("s",data_input.shape)
m, n = data_input.shape  # 数据形状
w = np.zeros(n)  # 训练参数
w = w.reshape(1, n)
y_ = np.ones(m, float)  # 预测输出


# data_input[0,0] = 33.36985
def sigmoid(inx):
    return 1.0 / (1.0 + np.exp(-inx))
cost_sum_show = []
cost_show = []
# data_input = np.insert(data_input, 0, values=1, axis=1)
cost_sum = 0
for i in range(1000):
    cost_sum = 0
    wt = w.transpose()
    theta = data_input * w
    theta_sum = np.sum(theta, 1)  # 算出了预测的Y，形成一个（m*1）的向量，其中 参数 1  为把第二维求和压缩维度为一维向量
    y_ = sigmoid(theta_sum)
    Q = y_ - data_label
    cost = np.sum( y_-data_label)
    temp = np.sum(data_input, 0,keepdims=True)
    temp_w = cost * np.sum(data_input, 0) * rate / m
    w -= cost * temp * rate / m
    cost_sum = np.sum(cost, 0)
    if i > 10:

        if i % 10 == 0:
            print(cost_sum)
            print(i)
            cost_sum_show.append(cost_sum)
            plt.plot(range(n), np.sum(w,0))
            plt.pause(0.001)
            plt.close()
    #if cost_sum < 0.0003:
     #   break
plt.plot(range(len(cost_sum_show)),cost_sum_show)

plt.show()