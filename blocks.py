from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout,
    Reshape
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Conv2DTranspose,
    Cropping2D
)
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import (Add, Multiply)
from keras import backend as K
import numpy as np
import h5py
from keras.optimizers import SGD, Adadelta, Adam
from keras.utils import np_utils
from keras import callbacks
from keras.callbacks import LearningRateScheduler,EarlyStopping, ModelCheckpoint
import math
import sys
from keras import losses
from keras import regularizers

K.set_image_dim_ordering('th')


if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
else:
    CHANNEL_AXIS = 1
    ROW_AXIS = 2
    COL_AXIS = 3

lrate = 0.00001

kernel_initializer = 'he_uniform'
bias_initializer = "zeros"
kernel_regularizer=regularizers.l2(0.0005)

def _residual_block(filters, kernel_size = 3, repetitions=1, padding = 'same'):
    def f(input):
        res1 = input
        for i in range(repetitions):
            res1 = _conv_relu(filters=filters,kernel_size=kernel_size, padding = padding)(res1)

        res1 = _conv(filters=filters,kernel_size=kernel_size, padding = padding)(res1)
        if padding=='valid':
            res2 = Cropping2D(cropping = ((kernel_size-1)//2)*(repetitions+1))(input)
        else:
            res2 = input

        res = Add()([res1, res2])

        return res

    return f

def _conv_block(filters, kernel_size = 3, subsample=(1, 1), padding = 'same'):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size = (kernel_size,kernel_size), subsample=subsample,
                             kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                             kernel_regularizer = kernel_regularizer, padding = padding)(input)
        norm = BatchNormalization(axis=CHANNEL_AXIS)(conv)
        return Activation("relu")(norm)

    return f

def _dense_block(node_arr):
    def f(input):
        temp = input
        for nodes in node_arr:
            temp = Dense(nodes,kernel_initializer=kernel_initializer,kernel_regularizer = kernel_regularizer)(temp)
            temp = BatchNormalization(axis=CHANNEL_AXIS)(temp)
            temp = Activation("relu")(temp)
        return temp

    return f

def _dense_relu(node_arr):
    def f(input):
        temp = input
        for nodes in node_arr:
            temp = Dense(nodes,kernel_initializer=kernel_initializer,kernel_regularizer = kernel_regularizer)(temp)
            # temp = BatchNormalization(axis=CHANNEL_AXIS)(temp)
            temp = Activation("relu")(temp)
        return temp

    return f

def _deconv_block(filters, kernel_size = 3, strides=(2, 2), padding = 'same'):
    def f(input):
        conv = Conv2DTranspose(filters=filters, kernel_size = (kernel_size,kernel_size), strides=strides,
                             kernel_initializer=kernel_initializer,bias_initializer=bias_initializer, 
                             kernel_regularizer = kernel_regularizer,padding = padding)(input)
        norm = BatchNormalization(axis=CHANNEL_AXIS)(conv)
        return Activation("relu")(norm)

    return f



def _conv_relu(filters, kernel_size = 3, subsample=(1, 1), padding = 'same'):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size = (kernel_size,kernel_size), subsample=subsample,
                             kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                             kernel_regularizer = kernel_regularizer, padding = padding)(input)
        active = Activation("relu")(conv)
        return active

    return f

def _conv_bn(filters, kernel_size = 3, subsample=(1, 1), padding = 'same'):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size = (kernel_size,kernel_size), subsample=subsample,
                             kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                             kernel_regularizer = kernel_regularizer, padding = padding)(input)
        norm = BatchNormalization(axis=CHANNEL_AXIS)(conv)
        return norm

    return f

def _conv(filters, kernel_size = 3, subsample=(1, 1), padding = 'same'):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size = (kernel_size,kernel_size), subsample=subsample,
                             kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                             kernel_regularizer = kernel_regularizer, padding = padding)(input)
        return conv

    return f