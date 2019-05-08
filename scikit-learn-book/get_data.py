# -*- coding: utf-8 -*-
# @Time    : 2019/4/26 3:34 PM
# @Author  : yangsheng
# @Email   : 891765948@qq.com
# @File    : get_data.py
# @Software: PyCharm
# @descprition:

import ssl
import os
import tarfile
from six.moves import urllib


ssl._create_default_https_context = ssl._create_unverified_context
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


fetch_housing_data()
