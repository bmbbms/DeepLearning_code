# -*- coding: utf-8 -*-
# @Time    : 2019/5/8 2:26 PM
# @Author  : yangsheng
# @Email   : 891765948@qq.com
# @File    : matplotusage.py
# @Software: PyCharm
# @descprition:


import numpy as np
import matplotlib.pyplot as plt





x = np.linspace(-5, 5, 20)
y = x ** 2 + 1
x_tricks = list()
for i in range(16):
    x_tricks.append(-5 + i)

plt.xticks(x_tricks)
plt.plot(x, y, "r:x", label="base")
plt.plot(x + 3, y, "b-D", label="moved")
plt.plot(x + 6, y, "y--_", label="big_moved")
plt.legend(loc="lower right")
plt.annotate("sample point",
             xy=(x[2], y[2]),
             xytext=(0, 22),
             arrowprops=dict(facecolor='black', shrink=0.05)
             )
# plt.show()
plt.title("sample of plt")
plt.xlabel("x", fontdict={"fontsize": 12}, )
plt.ylabel("y", fontdict={"fontsize": 12}, )
plt.savefig("a.png")
plt.show()