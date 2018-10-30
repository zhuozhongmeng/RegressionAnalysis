import numpy as np
import matplotlib.pyplot as plt
import math

rate = 0.000001
times = 10000000
data_np = np.loadtxt('data.txt')  # 用np来读区数据
q, t = data_np.shape
data_label = data_np[0:q, -1]  # 样本输出，取数据最后一列
data_input = data_np[0:q, 0:t - 1, ]  # 样本输入，去掉样本最后一列
one_insert = np.ones(q)
print(one_insert)
data_input = np.insert(data_input, 0, values=one_insert, axis=1)
print("s", data_input.shape)
m, n = data_input.shape  # 数据形状
w = np.zeros(n)  # 训练参数
w = w.reshape(1, n)
y_ = np.ones(m, float)  # 预测输出

data_label_np = data_label.reshape(m,1)
# data_input[0,0] = 33.36985
def sigmoid(inx):
    return 1.0 / (1.0 + np.exp(-inx))


cost_sum_show = []
cost_sum_show2 = []
cost_show = []
# data_input = np.insert(data_input, 0, values=1, axis=1)
cost_sum = 0

for i in range(times):
    cost_sum = 0
    theta_sum = np.dot(data_input, w.transpose())  # 矩阵乘积
    y_ = sigmoid(theta_sum)
    cost = np.sum(data_label_np - y_)
    temp = np.sum(data_input, 0, keepdims=True)
    w += cost * temp * rate / (m*67)
    cost_sum = np.sum(cost, 0)
    if i > 1000:
        if i % 10000 == 0:
            #print(cost_sum)
            print("进度：",i*100/times,"%")
            cost_sum_show2.append(cost_sum)
            # plt.plot(range(n), np.sum(w,0))
            # plt.pause(0.001)
            # plt.close()
plt.plot(range(len(cost_sum_show2)), cost_sum_show2, "r")
plt.show()
y_type  = np.ones(m)
for i in range(m):
    if y_[i] > 0.5:
        y_type[i] = 1
    else:
        y_type[i] = 0

plt.plot(range(m),data_label,)
plt.plot(range(m),y_type,)
plt.show()