import gzip
import logging
import struct
import numpy as np
import mxnet as mx

batch_size = 32

# data_dir = "/Users/yangsheng/py_code/Deeplearning_code/data/mnist"
logging.getLogger().setLevel(logging.DEBUG)
data_dir = "/Users/yangsheng/py_code/Deeplearning_code/data/fashion_data"


def read_data(label_url, image_url):
    with gzip.open(label_url) as label_file:
        magic, num = struct.unpack(">II", label_file.read(8))
        label = np.fromstring(label_file.read(), dtype=np.int8)
        print(label[0:10])

    with gzip.open(image_url, 'rb') as image_file:
        magic, num, rows, cols = struct.unpack(">IIII", image_file.read(16))
        image = np.fromstring(image_file.read(), dtype=np.uint8)
        image = image.reshape(len(label), 1, rows, cols)
        image = image.astype(np.float32) / 255.0
    return (label, image)


(t_label, t_image) = read_data(data_dir + "/train-labels-idx1-ubyte.gz", data_dir + "/train-images-idx3-ubyte.gz")
(val_label, val_image) = read_data(data_dir + "/t10k-labels-idx1-ubyte.gz", data_dir + "/t10k-images-idx3-ubyte.gz")

train_iter = mx.io.NDArrayIter(t_image, t_label, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(val_image, val_label, batch_size, shuffle=True)

data = mx.symbol.Variable('data')

# 第一个卷积层
conv1 = mx.sym.Convolution(data=data, name="conv1", kernel=(5, 5), num_filter=32)

# BN层批规范化
bn1 = mx.sym.BatchNorm(data=conv1, name="bn1", fix_gamma=False)

# 非线性激活层
act1 = mx.sym.Activation(data=bn1, name="act1", act_type="relu")

# 池化层
pool1 = mx.sym.Pooling(data=act1, name="pool1", pool_type="max", kernel=(3, 3), stride=(2, 2))

# 第二个卷积层 和BN层
conv2 = mx.sym.Convolution(data=pool1, name="conv2", kernel=(5, 5), num_filter=64)
bn2 = mx.sym.BatchNorm(data=conv2, name="bn2", fix_gamma=False)
# 非线性激活层
act2 = mx.sym.Activation(data=bn2, act_type="relu", name="act2")
# 第二个池化层
pool2 = mx.sym.Pooling(data=act2, name="pool2", pool_type="max", kernel=(3, 3), stride=(2, 2))

# 第三个卷积层相关
conv3 = mx.sym.Convolution(data=pool2, name="conv3", kernel=(3, 3), num_filter=10)
pool3 = mx.symbol.Pooling(data=conv3, kernel=(1, 1), pool_type="avg", name="pool3", global_pool=True)

flatten = mx.sym.Flatten(data=pool3, name="flatten")

net = mx.sym.SoftmaxOutput(data=flatten, name="softmax")

shape = {"data": (batch_size, 1, 28, 28)}
mx.viz.print_summary(symbol=net, shape=shape)

# mx.viz.plot_network(symbol=net, shape=shape).view()
mx.viz.plot_network(symbol=net, shape=shape).view()

mx.viz.plot_network(symbol=net, shape=shape).view()

module = mx.mod.Module(symbol=net)
# module = mx.mod.Module(symbol=net, context=mx.gpu(0))

module.fit(
    train_iter,
    eval_data=val_iter,
    optimizer='sgd',
    optimizer_params={"learning_rate": 0.2,
                      'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=60000 / batch_size, factor=0.9)},
    num_epoch=20,
    batch_end_callback=mx.callback.Speedometer(batch_size, 60000 / batch_size)
)
