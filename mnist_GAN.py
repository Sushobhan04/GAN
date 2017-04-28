<<<<<<< HEAD
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
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Conv2DTranspose,
    Cropping2D
)
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import numpy as np
import h5py
from keras.optimizers import SGD, Adadelta, Adam
from keras.utils import np_utils
from keras import callbacks
from keras.callbacks import LearningRateScheduler,EarlyStopping
import math
import sys
from keras.regularizers import l2
import blocks
import random
from keras.datasets import mnist
from keras.utils import plot_model

opt_d = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt_gan = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

def generator(nodes = 200, input_shape = 100):
# Build Generative model ...
    g_input = Input(shape=[input_shape])

    H = blocks._dense_block([nodes*14*14])(g_input)
    H = Reshape( [nodes, 14, 14] )(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = blocks._conv_block(nodes//2, kernel_size = 3)(H)
    H = blocks._conv_block(nodes//4, kernel_size = 3)(H)
    H = blocks._conv(1, kernel_size = 1)(H)
    g_V = Activation('sigmoid')(H)
    G = Model(g_input,g_V)

    print "Generated Generator"

    return G
    # generator.summary()

def discriminator(input_shape = (1,28,28), dropout_rate = 0.5, filters = 256):

    # Build Discriminative model ...
    d_input = Input(shape=input_shape)

    H = blocks._conv(filters, kernel_size = 5)(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = blocks._conv(filters, kernel_size = 5, subsample=(2, 2))(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(2,activation='softmax')(H)

    D = Model(d_input,d_V)

    print "Generated Discriminator"

    return D

def Generative_adversary(D,G,shape =100):
    # make_trainable(D,False)
    gan_input = Input(shape=[shape])
    H = G(gan_input)
    gan_V = D(H)
    GAN = Model(gan_input, gan_V)

    print "Generated GAN"

    return GAN

def create_training_data(samples = 100):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # print x_train.shape, y_train.shape
    x_train = x_train[:samples]/255.0

    label= np.zeros((samples,2))
    label[:,0] =1.0

    return x_train,label

# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
       l.trainable = val

def shuffle_pairs(data, label):
    c = list(zip(data, label))

    random.shuffle(c)

    data, label = zip(*c)

def train_D(D,G, trails = 100, epochs = 1):
    g_input = np.random.random((trails,100))
    g_data = G.predict(g_input)
    g_label = np.zeros((trails,2))
    g_label[:,1] = 1.0

    gt_data,gt_label = create_training_data(trails)


    train_data = np.concatenate((np.expand_dims(gt_data,axis = 1),g_data),axis=0)
    train_label = np.concatenate((gt_label,g_label), axis = 0)

    print np.max(train_data), np.max(train_label)

    shuffle_pairs(train_data, train_label)

    make_trainable(D,True)
    D.fit(train_data,train_label, epochs = epochs, verbose =1)

def train_GAN(GAN,G,D,trails=100, epochs = 10):
    g_input = np.random.random((trails,100))
    label = np.zeros((trails,2))
    label[:,0] = 1.0

    make_trainable(D,False)

    GAN = Generative_adversary(D,G, shape = 100)
    GAN.compile(loss='categorical_crossentropy', optimizer=opt_gan)

    GAN.fit(g_input,label,epochs = epochs, verbose =1)



def main():
    epochs = int(sys.argv[1])
    iterations = int(sys.argv[2])
    G = generator(nodes = 200, input_shape = 100)
    G.compile(loss='binary_crossentropy', optimizer=opt_gan)

    D = discriminator(input_shape = (1,28,28), dropout_rate = 0.5, filters = 256)
    D.compile(loss='categorical_crossentropy', optimizer=opt_d)

    GAN = Generative_adversary(D,G, shape = 100)
    GAN.compile(loss='categorical_crossentropy', optimizer=opt_gan)

    train_D(D,G, trails = 1000, epochs = 100)

    for i in range(iterations):
        print "Iteration :" + str(i)

        train_D(D,G, trails = 100, epochs = epochs)

        print 'D trained'

        train_GAN(GAN,G, D, trails=100, epochs = epochs)

        print 'G trained'


    GAN.save("GAN.h5")
    G.save("G.h5")
    D.save("D.h5")

    f = h5py.File('G.h5', 'r+')
    del f['optimizer_weights']
    f.close()

    plot_model(GAN, to_file='GAN.png', show_shapes = True)
    plot_model(G, to_file='G.png', show_shapes = True)
    plot_model(D, to_file='D.png', show_shapes = True)

if __name__ == '__main__':
    main()
=======
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
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import numpy as np
import h5py
from keras.optimizers import SGD, Adadelta, Adam
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
generator.compile(loss='binary_crossentropy', optimizer='adam')
# generator.summary()

input_g = np.random.random((nb_samples,100))

# Build Discriminative model ...
shp = (1,28,28)
dropout_rate = 0.5
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

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=opt)
# discriminator.summary()

output_g = generator.predict(input_g)
label_g = np.zeros((nb_samples,2))
label_g[:,1] = 1.0

print output_g.shape, label_g.shape 

input_train = np.concatenate((np.expand_dims(x_train,axis = 1),output_g),axis=0)
label_train = np.concatenate((label_t,label_g), axis = 0)

print input_train.shape, label_train.shape

discriminator.fit(input_train, label_train, nb_epoch = 100,verbose=1)

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
# GAN.summary()
GAN.save("GAN.h5")
>>>>>>> 2742eeafb9601b290e06119cbe5bb009f769c68d
