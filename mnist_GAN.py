from keras.models import (Model,Sequential, load_model)
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
import matplotlib.pyplot as plt
import pickle
import time

opt_d = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt_gan = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# opt_d_1 = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
batch_size = 128

def generator(nodes = 200, input_shape = 100):
# Build Generative model ...
    g_input = Input(shape=[input_shape])

    # H = blocks._dense_relu([1024])(H)
    H = blocks._dense_block([nodes*7*7])(g_input)
    H = Reshape( [nodes, 7, 7] )(H)
    H = blocks._conv_relu(nodes//2, kernel_size = 3)(H)
    H = blocks._deconv_block(nodes//2, kernel_size = 3)(H)
    H = blocks._conv_relu(nodes//4, kernel_size = 3)(H)
    H = blocks._deconv_block(nodes//4, kernel_size = 3)(H)
    H = blocks._conv(1, kernel_size = 3)(H)
    g_V = Activation('tanh')(H)
    G = Model(g_input,g_V)

    print "Generated Generator"

    return G
    # generator.summary()

def discriminator(input_shape = (1,28,28), dropout_rate = 0.25, filters = 256):

    # Build Discriminative model ...
    d_input = Input(shape=input_shape)
    H = blocks._conv(filters, kernel_size = 5)(d_input)

    H = blocks._conv(filters, kernel_size = 5,subsample=(2, 2))(d_input)
    H = LeakyReLU(0.2)(H)
    # H = Dropout(dropout_rate)(H)
    H = blocks._conv(filters, kernel_size = 5, subsample=(2, 2))(H)
    H = LeakyReLU(0.2)(H)
    # H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(1024)(H)
    H = LeakyReLU(0.2)(H)
    # H = Dropout(dropout_rate)(H)
    H = Dense(1)(H)
    d_V = Activation('sigmoid')(H)

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

def create_training_data():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # print x_train.shape, y_train.shape
    x_train = (x_train-127.0)/127.0
    # x_train = (x_train[:samples]-127.0)/127.0

    label= np.zeros((x_train.shape[0],1))
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

def train_D(D,G, true_data, trails = 100):
    g_input = np.random.uniform(-1.0,1.0,(trails,100))
    g_data = G.predict(g_input)
    g_label = np.zeros((trails,1))
    g_label[:,0] = 0.0

    gt_data = true_data[0]
    gt_label = true_data[1]

    train_data = np.concatenate((np.expand_dims(gt_data,axis = 1),g_data),axis=0)
    train_label = np.concatenate((gt_label,g_label), axis = 0)

    # print np.max(train_data), np.max(train_label)

    shuffle_pairs(train_data, train_label)

    D.trainable = True
    # D.compile(loss='binary_crossentropy', optimizer=opt_d)
    res = D.train_on_batch(train_data,train_label)
    return res
    # return res.history['loss'][0]


def train_GAN(GAN,D,trails=100):
    g_input = np.random.uniform(-1.0,1.0,(trails,100))
    label = np.zeros((trails,1))
    label[:,0] = 1.0
    # label[trails:,1] = 1.0

    shuffle_pairs(g_input, label)

    D.trainable = False
    # GAN.compile(loss='binary_crossentropy', optimizer=opt_gan)

    # GAN = Generative_adversary(D,G, shape = 100)
    # GAN.compile(loss='binary_crossentropy', optimizer=opt_gan)

    res = GAN.train_on_batch(g_input,label)
    # del GAN
    return res
    # return res.history['loss'][0]

def plot_loss(D_loss,GAN_loss, ylim= 5.0):
    plt.figure(figsize=(10,8))
    plt.plot(D_loss, label='discriminitive loss')
    plt.plot(GAN_loss, label='generative loss')
    plt.legend()
    plt.ylim((0,ylim))
    plt.savefig('output/GAN_plot.png')




def main():
    path_train = "/home/sushobhan/git/GAN/"
    iterations = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    model_name = sys.argv[3]

    gt_data,gt_label = create_training_data()

    G = generator(nodes = 200, input_shape = 100)
    G.compile(loss='binary_crossentropy', optimizer=opt_gan)

    # D = discriminator(input_shape = (1,28,28), dropout_rate = 0.5, filters = 256)
    # D.compile(loss='binary_crossentropy', optimizer=opt_d)

    D = load_model(path_train+'D_base.h5')

    GAN = Generative_adversary(D,G, shape = 100)
    GAN.compile(loss='binary_crossentropy', optimizer=opt_gan)

    # K.set_value(opt_d.lr, 1e-4)

    # beg = time.time()

    # for i in range(10):

    #     for j in range(gt_data.shape[0]//batch_size):

    #         d_loss = train_D(D,G,(gt_data[j*batch_size:(j+1)*batch_size],
    #             gt_label[j*batch_size:(j+1)*batch_size]),
    #              trails = batch_size)
    #         print i,j, d_loss
    # D.save("D_base.h5")

    # print 'time: ', time.time()-beg

    # K.set_value(opt_d.lr, 1e-5)

    D_loss = []
    GAN_loss =[]

    for i in range(iterations):
        # global opt_d, opt_gan
        # opt_d = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # opt_gan = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # if iterations>0:
        #     K.set_value(opt_d.lr, 1e-6)
        #     K.set_value(opt_gan.lr, 1e-4)
        # elif iterations>1:
        #     K.set_value(opt_d.lr, 1e-6)
        #     K.set_value(opt_gan.lr, 1e-5)
        # elif iterations>20:
        #     K.set_value(opt_d.lr, 1e-6)
        #     K.set_value(opt_gan.lr, 1e-5)

        print "Iteration :" + str(i)
        beg = time.time()

        for j in range(gt_data.shape[0]//batch_size):

            d_loss = train_D(D,G,(gt_data[j*batch_size:(j+1)*batch_size],
                gt_label[j*batch_size:(j+1)*batch_size]),
                 trails = batch_size)

            # D_loss.append(d_loss)

            gan_loss = train_GAN(GAN,D, trails=batch_size)

            print 'batch: ', str(j)
            print 'D loss: ', str(d_loss), ' GAN loss: ', str(gan_loss)

            # GAN_loss.append(gan_loss)

        print 'time: ', time.time()-beg
        D_loss.append(d_loss)
        GAN_loss.append(gan_loss)
        plot_loss(D_loss,GAN_loss,5.0)
        G.save('models/'+model_name+"G.h5")
        D.save('models/'+model_name+"D.h5")
        # K.clear_session()
        # G = load_model('models/'+model_name+'G.h5')
        # D = load_model('models/'+model_name+'D.h5')

        # if i%20 == 0:
        #     G.save('models/'+model_name+str(i)+"G.h5")
        #     D.save('models/'+model_name+str(i)+"D.h5")


    with open("test_"+model_name+".txt", "wb") as fp:
        pickle.dump(D_loss,fp)
        pickle.dump(GAN_loss,fp)



    # GAN.save("GAN.h5")
    # G.save("G.h5")
    # D.save("D.h5")

    # f = h5py.File('G.h5', 'r+')
    # del f['optimizer_weights']
    # f.close()

    # plot_model(GAN, to_file='GAN.png', show_shapes = True)
    # plot_model(G, to_file='G.png', show_shapes = True)
    # plot_model(D, to_file='D.png', show_shapes = True)

if __name__ == '__main__':
    main()
