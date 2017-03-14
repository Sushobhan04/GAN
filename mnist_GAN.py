from keras.models import (Model,Sequential)
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout,
    Reshape,
    Convolution2D,
    UpSampling2D
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
import h5py
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils
from keras import callbacks
from keras.callbacks import LearningRateScheduler,EarlyStopping
import math
import sys
from keras.regularizers import l2

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print x_train.shape, y_train.shape
nb_samples = 2000
x_train = x_train[:nb_samples,:,:]

label_t = np.zeros((nb_samples,2))
label_t[:,0] =1.0


# Build Generative model ...
nch = 200
g_input = Input(shape=[100])
H = Dense(nch*14*14, init='glorot_normal')(g_input)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Reshape( [nch, 14, 14] )(H)
H = UpSampling2D(size=(2, 2))(H)
H = Convolution2D(nch/2, 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(nch/4, 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
g_V = Activation('sigmoid')(H)
generator = Model(g_input,g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()

input_g = np.random.random((nb_samples,100))

# Build Discriminative model ...
shp = (1,28,28)
d_input = Input(shape=shp)
H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2,activation='softmax')(H)

discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()

output_g = generator.predict(input_g)
label_g = np.zeros((nb_samples,2))
label_g[:,1] = 1.0

input_train = np.concatenate((x_train,output_g),axis=0)
label_train = np.concatenate((label_t,label_g), axis = 0)

discriminator.fit(input_train, label_train, nb_epoch = 10,verbose=1)

# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
       l.trainable = val

make_trainable(discriminator, False)

# Build stacked GAN model
gan_input = Input(shape=[100])
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()