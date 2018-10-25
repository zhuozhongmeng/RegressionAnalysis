import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data2d.txt")
print(data)
times = 1000
rate = 0.0001
x=[]
y=[]
w0 = 0
w1 = 0
w2 = 0
for i in range(len(data)):
    if i < len(data)/2:
        x.append(data[i])
    if i >= len(data)/2:
        y.append(data[i])

plt.plot(x,y)
plt.pause(2)
plt.close()
plt.show()
delta_w0_sum =[]
delta_w1_sum =[]
delta_w2_sum =[]

for time in range(times):
    delta_w2 = 0
    delta_w1 = 0
    delta_w0 = 0
    y_=[]
    for i in range(len(x)):
        delta_w0 += w2*x[i]**2 +w1*x[i]+w0 - y[i]
        delta_w1 += (w2*x[i]**2 +w1*x[i]+w0- y[i])*x[i]
        delta_w2 += (w2*x[i]**2 +w1*x[i]+w0- y[i])*x[i]**2
        y_.append(w2*x[i]**2+w1*x[i]+w0)
    w2 -= rate*delta_w2/len(x)
    w1 -= rate*delta_w1/len(x)
    w0 -= rate*delta_w0/len(x)

    plt.plot(range(len(x)),y_)
    plt.plot(x,y,'x')
    plt.pause(0.001)
    plt.close()
    plt.show()
    y_ = []