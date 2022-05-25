import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, Input
from collections import namedtuple


def abs_criterion(pred, target):
    return tf.reduce_mean(tf.abs(pred - target))


def mae_criterion(pred, target):
    return tf.reduce_mean((pred - target) ** 2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def softmax_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def padding(x, p=3):
    return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

class InstanceNorm(layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def call(self, x):
        scale = tf.Variable(
            initial_value=np.random.normal(1., 0.02, x.shape[-1:]),
            trainable=True,
            name='SCALE',
            dtype=tf.float32
        )
        offset = tf.Variable(
            initial_value=np.zeros(x.shape[-1:]),
            trainable=True,
            name='OFFSET',
            dtype=tf.float32
        )
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return scale * normalized + offset


class ResNetBlock(layers.Layer):
    def __init__(self, dim, k_init, ks=3, s=1):
        super(ResNetBlock, self).__init__()
        self.dim = dim
        self.k_init = k_init
        self.ks = ks
        self.s = s
        self.p = (ks - 1) // 2
        # For ks = 3, p = 1
        self.padding = "valid"

    def call(self, x):
        y = layers.Lambda(padding, arguments={"p": self.p}, name="PADDING_1")(x)
        # After first padding, (batch * 130 * 130 * 3)

        y = layers.Conv2D(
            filters=self.dim,
            kernel_size=self.ks,
            strides=self.s,
            padding=self.padding,
            kernel_initializer=self.k_init,
            use_bias=False
        )(y)
        y = InstanceNorm()(y)
        y = layers.ReLU()(y)
        # After first conv2d, (batch * 128 * 128 * 3)

        y = layers.Lambda(padding, arguments={"p": self.p}, name="PADDING_2")(y)
        # After second padding, (batch * 130 * 130 * 3)

        y = layers.Conv2D(
            filters=self.dim,
            kernel_size=self.ks,
            strides=self.s,
            padding=self.padding,
            kernel_initializer=self.k_init,
            use_bias=False
        )(y)
        y = InstanceNorm()(y)
        y = layers.ReLU()(y + x)
        # After second conv2d, (batch * 128 * 128 * 3)

        return y


def build_discriminator(options, name='Discriminator'):
    initializer = tf.random_normal_initializer(0., 0.02)

    inputs = Input(shape=(options.time_step,
                          options.pitch_range,
                          options.output_nc))

    x = inputs

    x = layers.Conv2D(filters=options.df_dim,
                      kernel_size=7,
                      strides=2,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_1')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 32 * 42 * 64)

    x = layers.Conv2D(filters=options.df_dim * 4,
                      kernel_size=7,
                      strides=2,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_2')(x)
    x = InstanceNorm()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 16 * 21 * 256)

    x = layers.Conv2D(filters=1,
                      kernel_size=7,
                      strides=1,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_3')(x)
    # (batch * 16 * 21 * 1)

    #outputs = x
    x = layers.Flatten()(x)
    # (batch * 336)

    logits = layers.Dense(options.number_of_domains,name="Logits")(x)
    # (batch * number_of_domains)
    labels = layers.Softmax( name="Labels")(logits)
    # (batch * number_of_domains)
    outputs = labels


    return Model(inputs=inputs,
                 #outputs=[logits,labels] ,
                 outputs=[outputs],
                 name=name)


def build_generator(options, name='Generator'):
    initializer = tf.random_normal_initializer(0., 0.02)

    """
    inputs = Input(shape=(options.time_step,
                          options.pitch_range+options.number_of_domains,
                          options.output_nc))



    x = inputs
    # (batch * 64 * (84 + n_domains) * 1)
    x = layers.Flatten()(x)

    x = layers.Dense(options.pitch_range*options.time_step)(x)
    """
    inputs = Input(shape=(options.time_step * options.pitch_range * options.output_nc + options.number_of_domains))
    x = layers.Dense(options.time_step * options.pitch_range * options.output_nc)(inputs)
    x = layers.Reshape(target_shape=(options.time_step,
                                     options.pitch_range,
                                     options.output_nc))(x)
    # (batch * 64 * 84 * 1)

    x = layers.Lambda(padding,
                      name='PADDING_1')(x)
    # (batch * 70 * (90) * 1)

    x = layers.Conv2D(filters=options.gf_dim,
                      kernel_size=7,
                      strides=1,
                      padding='valid',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_1')(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)
    # (batch * 64 * 84 * 64)

    x = layers.Conv2D(filters=options.gf_dim * 2,
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_2')(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)
    # (batch * 32 * 42 * 128)

    x = layers.Conv2D(filters=options.gf_dim * 4,
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_3')(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)
    # (batch * 16 * 21 * 256)

    for i in range(10):
        # x = resnet_block(x, options.gf_dim * 4)
        x = ResNetBlock(dim=options.gf_dim * 4, k_init=initializer)(x)
    # (batch * 16 * 21 * 256)

    x = layers.Conv2DTranspose(filters=options.gf_dim * 2,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False,
                               name='DECONV2D_1')(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)
    # (batch * 32 * 42 * 128)

    x = layers.Conv2DTranspose(filters=options.gf_dim,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False,
                               name='DECONV2D_2')(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)
    # (batch * 64 * 84 * 64)

    x = layers.Lambda(padding,
                      name='PADDING_2')(x)
    # After padding, (batch * 70 * 90 * 64)

    x = layers.Conv2D(filters=options.output_nc,
                      kernel_size=7,
                      strides=1,
                      padding='valid',
                      kernel_initializer=initializer,
                      activation='sigmoid',
                      use_bias=False,
                      name='CONV2D_4')(x)
    # (batch * 64 * 84 * 1)

    outputs = x

    return Model(inputs=inputs,
                 outputs=outputs,
                 name=name)


def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.math.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.math.square(fake))

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if loss_func == 'hinge' :
        real_loss = tf.reduce_mean(relu(1.0 - real))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss

def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.math.squared_difference(fake, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge' :
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss

def classification_loss(logit, label) :
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))

    return loss

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss