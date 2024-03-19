"""
File:   frf_class_model.py
Auth:   C. DiMattina @ FGCU
Desc:   Keras model which works on the outputs of a set of fixed first-stage Gabor
        filters resembling V1 simple cells
"""
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from scipy.io import loadmat
import numpy as np

def my_init(shape, dtype):
    init = loadmat('./Filters/filters_concat_8_16_32.mat')['filter_mat']
    return tf.Variable(init,dtype=float)

class FRFClass(keras.Model):

    def __init__(self, num_hidden, pool_size, strides, l2, name="frf_class_model"):
        super(FRFClass, self).__init__(name=name)

        self.num_outputs    = 1
        self.filters        = 72
        self.kernel_size    = (32, 32)
        self.num_hidden     = num_hidden
        self.pool_size      = pool_size
        self.strides        = strides
        self.l2             = l2  # L2 weight penalty

        self.oneConv        = layers.Conv2D(self.filters, self.kernel_size, kernel_initializer=my_init,
                                    kernel_regularizer=tf.keras.regularizers.l2(l2), activation = "relu",
                                    use_bias =False, trainable=False, padding = "same", name = 'conv_layer')
        self.maxPool        = layers.MaxPool2D(pool_size=self.pool_size, strides=self.strides, name = 'max_pool_layer')
        self.hidden_dense   = layers.Dense(units=self.num_hidden , kernel_regularizer=tf.keras.regularizers.l2(l2),
                                           kernel_initializer = 'random_normal', activation = "relu", name='hidden_dense',
                                           use_bias=True)
        self.output_dense   = layers.Dense(units=self.num_outputs, kernel_initializer = 'random_normal', name='output_dense')


    @tf.function
    def call(self, inputs):
        x       = tf.expand_dims(inputs, axis=3)
        x       = self.oneConv(x)
        x       = self.maxPool(x)
        x       = tf.reshape(x, shape=(-1, x.shape[1]*x.shape[2]*x.shape[3]))
        x       = self.hidden_dense(x)
        out     = self.output_dense(x)
        return out

