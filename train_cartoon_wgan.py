from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import glob
import utils
import traceback
import numpy as np
import tensorflow as tf
import models_64x64 as models


""" param """
epoch = 100
batch_size = 35
lr = 0.0002
z_dim = 100
clip = 0.01
n_critic = 5
gpu_id = 0

''' data '''
# you should prepare your own data in ./data/faces
# cartoon faces original size is [96, 96, 3]

# 对图片做预处理，resize后做归一化，然后转换成tf的数据
def preprocess_fn(img):
    re_size = 64
    img = tf.to_float(tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
    return img

img_paths = glob.glob('./data/image/*.jpg')
data_pool = utils.DiskImageData(img_paths, batch_size, shape=[96, 96, 3], preprocess_fn=preprocess_fn)


""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    ''' models '''
    # 生成器
    generator = models.generator
    # 鉴别器
    discriminator = models.discriminator

    ''' graph '''
    # inputs
    # 定义输入的占位符，real是真实图片，z是噪声
    real = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    # generate
    # 生成的噪声，即假数据
    fake = generator(z, reuse=False)

    # dicriminate
    # 鉴别器对真实数据和假数据进行判别
    r_logit = discriminator(real, reuse=False)
    f_logit = discriminator(fake)

    # losses
    # wd = 真实图像的预测结果 - 假图像的预测结果
    # r_logit是真实图像的预测结果，所以r_logit的值越接近1，f_logit越接近0，就说明越准确，
    wd = tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit)
    # 这里使用差值构造了鉴别器的损失函数，即最小化d_loss = -wd，也就是要求r_logit的值最大化，f_logit最小化
    d_loss = -wd
    # 生成器的目标是最大化f_logit，与鉴别器相反，这里体现了对抗的思想
    g_loss = -tf.reduce_mean(f_logit)

    # otpims
    # d_var可以理解为需要优化的参数，或者说这个命名空间里的参数
    d_var = utils.trainable_variables('discriminator')
    g_var = utils.trainable_variables('generator')
    # 鉴别器的优化，使用RMSPropOptimizer优化器，（有时间试试ranger优化器，就是不确定有没有tf1.x的实现代码）
    d_step_ = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(d_loss, var_list=d_var)
    # tf.control_dependencies：此函数指定某些操作执行的依赖关系，具体没研究
    with tf.control_dependencies([d_step_]):
        d_step = tf.group(*(tf.assign(var, tf.clip_by_value(var, -clip, clip)) for var in d_var))
    # 生成器的优化
    g_step = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(g_loss, var_list=g_var)

    # summaries
    # 目测是为了使用tensorboard进行可视化
    d_summary = utils.summary({wd: 'wd'})
    g_summary = utils.summary({g_loss: 'g_loss'})

    # sample
    # 生成样本？
    f_sample = generator(z, training=False)


""" train """
''' init '''
# session
sess = utils.session()
# iteration counter
it_cnt, update_cnt = utils.counter()
# saver
# 保留最近的 5 个检查点文件
saver = tf.train.Saver(max_to_keep=5)
# summary writer
summary_writer = tf.summary.FileWriter('./summaries/cartoon_wgan', sess.graph)

''' initialization '''
# 检查checkpiont，没有就创建，为了断点保存，方便继续训练啥的
ckpt_dir = './checkpoints/cartoon_wgan'
utils.mkdir(ckpt_dir + '/')
# 导入之前训练的模型，没有就从头开始训练
if not utils.load_checkpoint(ckpt_dir, sess):
    sess.run(tf.global_variables_initializer())

''' train '''
try:
    # 生成一个[100, z_dim]随机的正态分布噪声
    z_ipt_sample = np.random.normal(size=[100, z_dim])
    # 计算每个epoch有多少个batch
    batch_epoch = len(data_pool) // (batch_size * n_critic)
    # 总的batch数量，在下面的for循环使用
    max_it = epoch * batch_epoch

    for it in range(sess.run(it_cnt), max_it):
        # 更新当前的epoch数，计数用
        sess.run(update_cnt)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        # train D
        # 这个if没看懂要干嘛，
        if it < 25:
            c_iter = 100
        else:
            c_iter = n_critic
        # 先训练c_iter次鉴别器
        for i in range(c_iter):
            # batch data
            # 获取一个批次的图像数据real_ipt和噪音数据z_ipt作为输入
            real_ipt = data_pool.batch()
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            # 把真实图像和噪声喂入网络，更新d_step，即训练鉴别器
            d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z: z_ipt})
        # 这个应该是tensorboard可视化什么的东西
        summary_writer.add_summary(d_summary_opt, it)

        # train G
        # 输入噪声，更新g_step，即训练生成器
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt})
        # 这个应该是tensorboard可视化什么的东西
        summary_writer.add_summary(g_summary_opt, it)

        # display
        # 每十次输出一次
        if it % 10 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        # 500次保存一次模型
        if (it + 1) % 500 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        # 每100保存生成图像样本，
        if (it + 1) % 100 == 0:
            # 输入z_ipt_sample，产生样本
            f_sample_opt = sess.run(f_sample, feed_dict={z: z_ipt_sample})

            save_dir = './sample_images_while_training/cartoon_wgan'
            utils.mkdir(save_dir + '/')
            utils.imwrite(utils.immerge(f_sample_opt, 10, 10), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))

#如果报错输出报错信息
except Exception as e:
    traceback.print_exc()

# 训练结束，关闭会话
finally:
    print(" [*] Close main session!")
    sess.close()
