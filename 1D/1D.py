import numpy as np
import matplotlib.pyplot as plt
import time

data = np.loadtxt("data.txt")
print(data)
times = 30
rate = 0.0001
x = []
y = []
y_ = []
delta_sum = []
delta_sum_temp = 0
w = 0
b = 0

print(len(data))
for i in range(len(data)):
    if i % 2 == 0:
        x.append(data[i])
    if i % 2 == 1:
        y.append(data[i])

plt.plot(x, y)
plt.pause(1)
plt.close()
plt.show()
delta_b_sum = []
for time in range(times):
    delta_w = 0
    delta_b = 0
    for i in range(len(x)):
        # print(i)
        delta_w += (w * x[i] + b - y[i]) * x[i]
        delta_b += w * x[i] + b - y[i]
        y_.append(w * x[i] + b)
        delta_sum_temp += (w * x[i] + b - y[i]) ** 2
    delta_sum.append(delta_sum_temp)
    delta_sum_temp = 0
    w -= rate * delta_w / len(x)
    b -= rate * delta_b / len(x)
    delta_b_sum.append(b)
    if time % 10 == 0:
        plt.plot(x, y_)
        plt.plot(x, y)
        plt.pause(0.01)
        plt.close()
        plt.show()
        print(time)
    y_ = []

plt.plot(range(times), delta_sum)
plt.plot(range(times), delta_b_sum)
plt.show()
plt.pause(1)
print(type(y_))
File = open("w.txt", "w+")
for i in y_:
    File.write(i)
    print(i)
    File.write('q')
    File.flush()
File.close()