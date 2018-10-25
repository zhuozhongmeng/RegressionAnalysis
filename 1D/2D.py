import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data2d.txt")
print(data)
times = 3000
rate = 0.00001
x=[]
y=[]
w0 = 0
w1 = 0
w2 = 0
b = -2
for i in range(len(data)):
    if i < len(data)/2:
        x.append(data[i])
    if i >= len(data)/2:
        y.append(data[i])

plt.plot(x,y)
plt.pause(0.2)
plt.close()
plt.show()
delta_w0_sum =[]
delta_w1_sum =[]
delta_w2_sum =[]
delta_b_sum=[]

for time in range(times):
    delta_w2 = 0
    delta_w1 = 0
    delta_w0 = 0
    delta_b = 0
    y_=[]
    for i in range(len(x)):
        delta_w0 += w2*(x[i]+b)**2 +w1*(x[i]+b)+w0 - y[i]
        delta_w1 += (w2*(x[i]+b)**2 +w1*(x[i]+b)+w0- y[i])*(x[i]+b)
        delta_w2 += (w2*(x[i]+b)**2 +w1*(x[i]+b)+w0- y[i])*(x[i]+b)**2
        delta_b += (w2*(x[i]+b)**2 +w1*(x[i]+b)+w0- y[i])*(2*w2*(x[i]+b)+w1)
        y_.append(w2*(x[i]+b)**2+w1*(x[i]+b)+w0)
    w2 -= rate*delta_w2/len(x)
    w1 -= rate*delta_w1/len(x)
    w0 -= rate*delta_w0/len(x)
    b -= rate*delta_b/len(x)
    delta_b_sum.append(delta_b)
    delta_w0_sum.append(delta_w0)
    delta_w1_sum.append(delta_w1)
    delta_w2_sum.append(delta_w2)
    if time %10 == 0:
        plt.plot(x,y_)
        plt.plot(x,y,'x')
        print(b)
        plt.pause(0.001)
        plt.close()
        plt.show()
        y_ = []
plt.plot(range(times), delta_w2_sum)
plt.plot(range(times), delta_w1_sum)
plt.plot(range(times), delta_w0_sum)
plt.plot(range(times), delta_b_sum ,"y")
plt.show()