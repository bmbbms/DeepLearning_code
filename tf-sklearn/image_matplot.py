# -*- coding: utf-8 -*-
# @Time    : 2019/5/8 3:20 PM
# @Author  : yangsheng
# @Email   : 891765948@qq.com
# @File    : image_matplot.py
# @Software: PyCharm
# @descprition:

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


img = mpimg.imread("a.png")

print(len(img))
print(img.shape)

grey_img = img[:, :, 0]
print(grey_img[200:201])
print(grey_img.shape)
import scipy
print(scipy.__version__)


plt.imshow(grey_img, cmap="binary")
plt.show()