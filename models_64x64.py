from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim

from functools import partial

# partial(f,x):把x作为参数传给f函数
# 卷积
conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
#反卷积
dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
# 全连接
fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)
# 批归一化
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
ln = slim.layer_norm

# 生成器
def generator(z, dim=64, reuse=True, training=True):

    # 相当于定义了bn,dconv_bn_relu,fc_bn_relu函数
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    # 在变量空间中操作，好像也方便使用tensorboard
    with tf.variable_scope('generator', reuse=reuse):
        # 把噪声通过全连接增加维度，激活函数前还使用了bn
        y = fc_bn_relu(z, 4 * 4 * dim * 8)
        # reshape成特征图的形式，为了下面的卷积操作
        y = tf.reshape(y, [-1, 4, 4, dim * 8])

        # 4次反卷积，增大特征图尺寸到生成的图片大小
        y = dconv_bn_relu(y, dim * 4, 5, 2)
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        y = dconv_bn_relu(y, dim * 1, 5, 2)
        img = tf.tanh(dconv(y, 3, 5, 2))
        return img

# 鉴别器网络
def discriminator(img, dim=64, reuse=True, training=True):
    # 定义bn，conv_bn_lrelu操作
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('discriminator', reuse=reuse):
        # 使用卷积进行提取特征，后面还用了bn操作
        y = lrelu(conv(img, dim, 5, 2))
        y = conv_bn_lrelu(y, dim * 2, 5, 2)
        y = conv_bn_lrelu(y, dim * 4, 5, 2)
        y = conv_bn_lrelu(y, dim * 8, 5, 2)
        # 输出一维，作为判断结果，目测是二分类问题
        logit = fc(y, 1)
        return logit


def discriminator_wgan_gp(img, dim=64, reuse=True, training=True):
    conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('discriminator', reuse=reuse):
        y = lrelu(conv(img, dim, 5, 2))
        y = conv_ln_lrelu(y, dim * 2, 5, 2)
        y = conv_ln_lrelu(y, dim * 4, 5, 2)
        y = conv_ln_lrelu(y, dim * 8, 5, 2)
        logit = fc(y, 1)
        return logit
