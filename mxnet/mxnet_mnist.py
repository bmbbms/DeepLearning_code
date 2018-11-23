import numpy as np
import os
import gzip
import struct
import logging
import mxnet as mx
import matplotlib.pyplot as plt

data_dir = "/Users/yangsheng/py_code/Deeplearning_code/data/mnist"

logging.getLogger().setLevel(logging.DEBUG)

def read_data(label_url,image_url):
    with gzip.open(label_url) as label_file:
        magic,num = struct.unpack(">II", label_file.read(8))
        label = np.fromstring(label_file.read(),dtype=np.int8)
        print(label[0:10])

    with gzip.open(image_url, 'rb') as image_file:
        magic,num,rows,cols = struct.unpack(">IIII", image_file.read(16))
        image = np.fromstring(image_file.read(), dtype=np.uint8)
        image = image.reshape(len(label),1,rows,cols)
        image = image.astype(np.float32)/255.0
    return (label,image)

(t_label,t_image) = read_data(data_dir+"/train-labels-idx1-ubyte.gz",data_dir+"/train-images-idx3-ubyte.gz")
(val_label,val_image) = read_data(data_dir+"/t10k-labels-idx1-ubyte.gz",data_dir+"/t10k-images-idx3-ubyte.gz")

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(t_image[i].reshape(28,28),cmap="Greys_r")
    plt.axis("off")
plt.show()
print('label:%s' % (t_label[0:10]))

batch_size = 32
# 迭代器
train_iter = mx.io.NDArrayIter(t_image,t_label,batch_size,shuffle=True)
val_iter = mx.io.NDArrayIter(val_image,val_label,batch_size,shuffle=True)

data = mx.symbol.Variable("data")

flatten = mx.sym.Flatten(data=data, name="flatten")

fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=128, name='fc1')
act1 = mx.sym.Activation(data=fc1,act_type="relu", name="act1")


fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64, name='fc2')
act2 = mx.sym.Activation(data=fc2,act_type="relu", name="act2")

fc3 = mx.sym.FullyConnected(data=act2,num_hidden=10,name="fc3")
net  = mx.sym.SoftmaxOutput(data=fc3,name="softmax")

shape = {"data": (batch_size,1,28,28)}
mx.viz.print_summary(symbol=net,shape=shape)

# mx.viz.plot_network(symbol=net, shape=shape).view()
mx.viz.plot_network(symbol=net, shape=shape).view()

module = mx.mod.Module(symbol=net)
module = mx.mod.Module(symbol=net, context=mx.gpu(0))

module.fit(
    train_iter,
    eval_data=val_iter,
    optimizer='sgd',
    optimizer_params={"learning_rate":0.2, 'lr_scheduler':mx.lr_scheduler.FactorScheduler(step=60000/batch_size,factor=0.9)},
    num_epoch=20,
    batch_end_callback = mx.callback.Speedometer(batch_size,60000/batch_size)
)