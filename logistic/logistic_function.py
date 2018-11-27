import numpy as np
import matplotlib.pyplot as plt
import function as fc
import time

time_start = time.clock()
data = np.loadtxt("datasoftmax.txt")
data_test = np.loadtxt("datatest.txt")
# data_x_0, data_y_0, data_x_1, data_y_1 = fc.m_getdata(data, True, data_y_shape = 3)
data_x, data_y = fc.m_getdata(data, data_y_shape=3)
#先定义一下数据
m, n = data_x.shape
p, q = data_y.shape
w = np.ones([n, q])
HH= fc.m_r_softmax(data_x,data_y,w)
print(HH)
data_x_test ,data_y_test = fc.m_getdata(data_test,data_y_shape = 3)
Ht =  fc.m_softmax_test(data_x,data_y,HH)
# data_x_0 = np.array(data_x_0)
# data_x_1 = np.array(data_x_1)
# plt.plot(data_x_0[:, 0], data_x_0[:, 1], 'x')
# plt.plot(data_x_1[:, 0], data_x_1[:, 1], 'o')
# plt.show()

# w = fc.m_logistic(data_x, data_y, 0, 0.0001, 30000)
# time_stop = time.clock()
# print("计算用时", time_stop - time_start, "秒")

# data_x_test, data_y_test = fc.m_getdata(data_test)

# fc.m_test_w(w, data_x_test, data_y_test)
