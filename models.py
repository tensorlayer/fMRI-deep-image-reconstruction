import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS


def encoder(x, resblk=4, z_dim=100, is_train=True, reuse=False):
    w_init = tf.truncated_normal_initializer(stddev=0.02)
    g_init = tf.truncated_normal_initializer(mean=1.0, stddev=0.02)
    b_init = None

    gf_dim = 64
    filters = (3, 3)
    stride = (2, 2)

    with tf.variable_scope('encoder', reuse=reuse):

        # (64, 64, 3)
        net_in = InputLayer(x, name='e/in')

        # (32, 32, 64)
        net = Conv2d(net_in, n_filter=gf_dim, filter_size=filters, strides=stride,
                     act=tf.nn.relu, padding='SAME', W_init=w_init, name='e/n32s2/c0')

        # (16, 16, 128)
        net = Conv2d(net, n_filter=gf_dim * 2, filter_size=filters, strides=stride,
                     act=tf.nn.relu, padding='SAME', W_init=w_init, name='e/n64s2/c0')

        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                             name='e/n64s2/b0')

        # (8, 8, 256)
        net = Conv2d(net, n_filter=gf_dim * 4, filter_size=filters, strides=stride,
                     act=tf.nn.relu, padding='SAME', W_init=w_init, name='e/n128s2/c0')

        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                             name='e/n128s2/b0')

        # Residual Blocks
        """temp = net

        res_filter = (3, 3)
        res_stride = (2, 2)

        for i in range(resblk):
            net_r = Conv2d(net, n_filter=gf_dim * 4, filter_size=res_filter, strides=res_stride, act=tf.identity,
                           padding='SAME', W_init=w_init, b_init=b_init, name='e/n256s1/c1/%s' % i)

            net_r = BatchNormLayer(net_r, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='e/n64s1/b1/%s' % i)

            net_r = Conv2d(net_r, n_filter=gf_dim * 4, filter_size=res_filter, strides=res_stride, act=tf.identity,
                           padding='SAME', W_init=w_init, b_init=b_init, name='e/n256s1/c2/%s' % i)

            net_r = BatchNormLayer(net_r, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='e/n64s1/b2/%s' % i)

            net_r = ElementwiseLayer([net, net_r], combine_fn=tf.add, name='e/residual_add/%s' % i)

            net = net_r

        net = Conv2d(net, n_filter=gf_dim * 4, filter_size=res_filter, strides=res_stride, act=tf.identity,
                     W_init=w_init, b_init=b_init, name='e/n256s1/c/m')

        net = BatchNormLayer(net, is_train=is_train, gamma_init=g_init, name='e/n256s1/b/m')

        net = ElementwiseLayer([net, temp], combine_fn=tf.add, name='e/add3')"""

        # End of Residual Blocks

        net = FlattenLayer(net, name='e/flatten')
        out = DenseLayer(net, n_units=z_dim, act=tf.identity, W_init=w_init, name='e/lin/output')

    return out, out.outputs


def generator(z, y, h_dim=128, is_train=True, reuse=False):
    w_init = tf.truncated_normal_initializer(stddev=0.02)
    g_init = tf.truncated_normal_initializer(mean=1.0, stddev=0.02)

    gf_dim = 64
    c_dim = 3
    filters = (3, 3)
    stride = (2, 2)

    image_size = 64
    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16),

    # Conditional GAN, adds y value to input layer
    feat_vec = tf.concat(axis=1, values=[z, y])
    # Ensures current batch size matches
    batch_size = feat_vec.get_shape().as_list()[0]

    with tf.variable_scope('generator', reuse=reuse):

        # (100, )
        net_in = InputLayer(feat_vec, name='g/in')

        # (128, )
        net_in = DenseLayer(net_in, n_units=h_dim, act=tf.identity,
                            W_init=w_init, name='g/h0/lin0')
        # (64*16*4*4 = 4*4*1024)
        net_h0 = DenseLayer(net_in, n_units=gf_dim*16*s16*s16, W_init=w_init,
                            act=tf.identity, name='g/h0/lin1')

        # (4, 4, 1024)
        net_h0 = ReshapeLayer(net_h0, shape=(-1, s16, s16, gf_dim*16), name="g/h0/reshape")

        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g/h0/batch_norm')

        # (8, 8, 512)
        net_h1 = DeConv2d(net_h0, n_filter=gf_dim * 8, filter_size=filters, out_size=(s8, s8), strides=stride,
                          padding='SAME', batch_size=batch_size, W_init=w_init, name='g/h1/deconv2d')

        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g/h1/batch_norm')

        # (16, 16, 256)
        net_h2 = DeConv2d(net_h1, n_filter=gf_dim * 4, filter_size=filters, out_size=(s4, s4), strides=stride,
                          padding='SAME', batch_size=batch_size, W_init=w_init, name='g/h2/deconv2d')

        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g/h2/batch_norm')

        # (32, 32, 128)
        net_h3 = DeConv2d(net_h2, n_filter=gf_dim * 2, filter_size=filters, out_size=(s2, s2), strides=stride,
                          padding='SAME', batch_size=batch_size, W_init=w_init, name='g/h3/deconv2d')

        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='g/h3/batch_norm')

        # (64, 64, 3)
        out = DeConv2d(net_h3, n_filter=c_dim, filter_size=filters, out_size=(image_size, image_size), strides=stride,
                       padding='SAME', batch_size=batch_size, W_init=w_init, name='g/h4/deconv2d')

        out.outputs = tf.nn.tanh(out.outputs)

        return out, out.outputs


def discriminator(x_input, is_train=True, reuse=False):
    w_init = tf.truncated_normal_initializer(stddev=0.02)
    gamma_init = tf.truncated_normal_initializer(mean=1.0, stddev=0.02)

    df_dim = 64
    filters = (5, 5)
    stride = (2, 2)

    with tf.variable_scope('discriminator', reuse=reuse):

        # (64, 64, 3)
        net_in = InputLayer(inputs=x_input, name='d/in')

        # (32, 32, 64)
        net_h0 = Conv2d(net_in, n_filter=df_dim, filter_size=filters, strides=stride,
                        act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='d/h0/conv2d')

        # (16, 16, 128)
        net_h1 = Conv2d(net_h0, n_filter=df_dim * 2, filter_size=filters, strides=stride,
                        act=None, padding='SAME', W_init=w_init, name='d/h1/conv2d')

        net_h1 = BatchNormLayer(net_h1, act=tf.nn.leaky_relu, is_train=is_train,
                                gamma_init=gamma_init, name='d/h1/batch_norm')

        # (8, 8, 256)
        net_h2 = Conv2d(net_h1, n_filter=df_dim * 4, filter_size=filters, strides=stride,
                        act=None, padding='SAME', W_init=w_init, name='d/h2/conv2')

        net_h2 = BatchNormLayer(net_h2, act=tf.nn.leaky_relu, is_train=is_train,
                                gamma_init=gamma_init, name='d/h2/batch_norm')

        # (4, 4, 512)
        net_h3 = Conv2d(net_h2, n_filter=df_dim * 8, filter_size=filters, strides=stride,
                        act=None, padding='SAME', W_init=w_init, name='d/h3/conv2')

        net_h3 = BatchNormLayer(net_h3, act=tf.nn.leaky_relu, is_train=is_train,
                                gamma_init=gamma_init, name='d/h3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='d/h4/flatten')
        out = DenseLayer(net_h4, n_units=1, act=tf.identity, W_init=w_init, name='d/h4/lin_sigmoid')

    return out, out.outputs


def code_discriminator(z, reuse=False):
    w_init = tf.truncated_normal_initializer(stddev=0.02)

    h = 4096

    with tf.variable_scope('code_evaluator', reuse=reuse):

        net_in = InputLayer(z, name='cd/in')

        net = DenseLayer(net_in, n_units=h, act=tf.nn.leaky_relu, W_init=w_init, name="cd/fc1")

        net = DenseLayer(net, n_units=h, act=tf.nn.leaky_relu, W_init=w_init, name="cd/fc2")

        out = DenseLayer(net, n_units=1, act=tf.identity, W_init=w_init, name="cd/sigmoid")

    return out, out.outputs
