import numpy as np
import matplotlib.pyplot as plt
import math

# Leslie的Triangle2学习率衰减方法
def Triangular2(T_max, gamma):
    def new_lr(step):
        region = step // T_max + 1    # 所处分段
        increase_rate = 1 / T_max * math.pow(gamma, region - 2)    # 增长率的绝对值
        return increase_rate * (step - (region - 1) * T_max) if step <= (region - 0.5) * T_max else - increase_rate * (step - region * T_max)
    return new_lr

#  Loshchilov＆Hutter 提出的warm_start
def WarmStart(T_max):
    def new_lr(step):
        return math.cos((step % T_max) * math.pi / T_max) + 1
    return new_lr

# 获取映射
Lambda_lr = WarmStart(T_max=100)
x = np.linspace(1, 600, 1000)
y = np.array([Lambda_lr(num) for num in x])
plt.plot(x, y)
plt.show()
