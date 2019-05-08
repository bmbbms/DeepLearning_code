# -*- coding: utf-8 -*-
# @Time    : 2019/5/8 4:16 PM
# @Author  : yangsheng
# @Email   : 891765948@qq.com
# @File    : linear_model.py
# @Software: PyCharm
# @descprition:
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

"""
x = np.array([[0, 1], [3, -2], [2, 3]])
y = np.array([0.5, 0.3, 0.9])

reg = linear_model.LinearRegression()

reg.fit(x, y)

print(reg.intercept_)
print(reg.coef_)
print(reg.predict([[1, 2], [4, 3], [0, 1]]))

np.vstack()

"""


def make_data(ndim):
    x0 = np.linspace(1, np.pi, 50)
    x = np.vstack([[x0, ], [i ** x0 for i in range(2, ndim + 1)]])
    y = np.sin(x0)+np.random.normal(0,0.15, len(x0))
    return x.transpose(), y


def linear_reg(x, y):
    dims = [1, 3, 6, 12]

    for index, i in enumerate(dims):
        plt.subplot(2, len(dims)/2, index+1)
        reg = linear_model.LinearRegression()
        sub_x = x[:, 0:i]
        reg.fit(sub_x, y)
        plt.plot(x[:, 0], reg.predict(sub_x))
        plt.plot(x[:, 0], y)
        plt.title("dim=%s" % i)

        print("dim %d" %i)
        print("intercept_ :%s" % (reg.intercept_))
        print("coef_ :%s" % (reg.coef_))
    plt.show()


def linear_reg_ling(x, y):
    # dims = [1, 3, 6, 12]
    alpha = [1e-15, 1e-12, 1e-5, 1,]
    for index, i in enumerate(alpha):
        plt.subplot(2, len(alpha)/2, index+1)
        reg = linear_model.Ridge(alpha=i)

        reg.fit(x, y)
        plt.plot(x[:, 0], reg.predict(x))
        plt.plot(x[:, 0], y)
        plt.title("dim=12,alpha=%e" % i)

        print("alpha  %e" %i)
        print("intercept_ :%s" % (reg.intercept_))
        print("coef_ :%s" % (reg.coef_))
    plt.show()

def linear_reg_lasso(x, y):
    # dims = [1, 3, 6, 12]
    alpha = [1e-10, 1e-3, 1, 10,]
    for index, i in enumerate(alpha):
        plt.subplot(2, len(alpha)/2, index+1)
        reg = linear_model.Lasso(alpha=i)

        reg.fit(x, y)
        plt.plot(x[:, 0], reg.predict(x))
        plt.plot(x[:, 0], y, ".")
        plt.title("dim=12,alpha=%e" % i)

        print("alpha  %e" % i)
        print("intercept_ :%s" % (reg.intercept_))
        print("coef_ :%s" % (reg.coef_))
    plt.show()


if __name__ == "__main__":
    x, y = make_data(12)
    # linear_reg(x, y)
    # linear_reg_ling(x, y)
    linear_reg_lasso(x, y)