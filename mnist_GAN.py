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
import tensorflow as tf
import argparse
from keras.models import load_model
import cv2
import os


K.set_image_dim_ordering('th')
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

opt_d = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt_gan = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)

# opt_d = SGD(lr=0.0005, momentum=0.9, nesterov=True)
# opt_gan = SGD(lr=0.0005, momentum=0.9, nesterov=True)

# opt_d_1 = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default='model1')
    parser.set_defaults(pretty=False)
    args = parser.parse_args()
    return args

def generator(nodes = 200, input_shape = 100):
# Build Generative model ...
    g_input = Input(shape=[input_shape])

    G = Dense(nodes)(g_input)
    G = Activation('tanh')(G)

    G = Dense(128*7*7)(G)
    G = BatchNormalization()(G)
    G = Activation('tanh')(G)

    G = Reshape( [128, 7, 7] )(G)
    G = UpSampling2D(size=(2,2))(G)

    G = Convolution2D(64,5,5, border_mode='same')(G)
    G = Activation('tanh')(G)

    G = UpSampling2D(size=(2,2))(G)

    G = Convolution2D(1,5,5, border_mode='same')(G)
    G = Activation('tanh')(G)

    G = Model(g_input,G)

    return G

    # model = Sequential()
    # model.add(Dense(input_dim=100, output_dim=1024))
    # model.add(Activation('tanh'))
    # model.add(Dense(128*7*7))
    # model.add(BatchNormalization())
    # model.add(Activation('tanh'))
    # model.add(Reshape((128,7,7), input_shape=(128*7*7,)))
    # model.add(UpSampling2D(size=(2,2), dim_ordering="th"))
    # model.add(Convolution2D(64,5,5, border_mode='same', dim_ordering="th"))
    # model.add(Activation('tanh'))
    # model.add(UpSampling2D(size=(2,2), dim_ordering="th"))
    # model.add(Convolution2D(1,5,5, border_mode='same', dim_ordering="th"))
    # model.add(Activation('tanh'))

    # print "Generated Generator"

    # return model
    # generator.summary()

def discriminator(input_shape = (1,28,28), dropout_rate = 0.25, filters = 256):

    d_input = Input(shape=input_shape)

    D = Convolution2D(64,5,5, border_mode='same')(d_input)
    D = Activation('tanh')(D)

    D = MaxPooling2D(pool_size=(2,2))(D)

    D = Convolution2D(128,5,5, border_mode='same')(D)
    D = Activation('tanh')(D)

    D = Flatten()(D)

    D = Dense(1024)(D)
    D = Activation('tanh')(D)

    D = Dense(1)(D)
    D = Activation('sigmoid')(D)

    D = Model(d_input,D)

    return D



    # model = Sequential()
    # model.add(Convolution2D(64,5,5,
    #                         border_mode='same',
    #                         input_shape=(1,28,28),
    #                         dim_ordering="th"))
    # model.add(Activation('tanh'))
    # model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
    # model.add(Convolution2D(128,5,5, border_mode='same', dim_ordering="th"))
    # model.add(Activation('tanh'))
    # model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
    # model.add(Flatten())
    # model.add(Dense(1024))
    # model.add(Activation('tanh'))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    # return model

def Generative_adversary(D,G,shape =100):
    # make_trainable(D,False)
    model = Sequential()
    model.add(G)
    D.trainable = False
    model.add(D)
    return model

def create_training_data():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # print x_train.shape, y_train.shape
    x_train = (x_train-127.5)/127.5
    # x_train = (x_train[:samples]-127.0)/127.0

    label= np.zeros(x_train.shape[0])
    label[:] =1.0

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

def create_random(batch_size =128, val = 0):
    g_input = np.zeros((batch_size, 100))
    for i in range(batch_size):
        g_input[i,:] = np.random.uniform(-1.0,1.0,100)
    g_label = np.zeros(batch_size)
    g_label[:] = val

    return g_input,g_label

def train_D(D,G, true_data, batch_size = 128):
    g_data,g_label = create_random(batch_size)

    g_data = G.predict(g_data)

    gt_data = true_data[0]
    gt_label = true_data[1]

    train_data = np.concatenate((np.expand_dims(gt_data,axis = 1),g_data),axis=0)
    train_label = np.concatenate((gt_label,g_label), axis = 0)

    # shuffle_pairs(train_data, train_label)

    D.trainable = True
    res = D.train_on_batch(train_data,train_label)
    return res


def train_GAN(GAN,D,batch_size=128):
    g_input,g_label = create_random(100,1)

    D.trainable = False

    res = GAN.train_on_batch(g_input,g_label)
    return res

def plot_loss(D_loss,GAN_loss, ylim= 5.0):
    plt.figure(figsize=(10,8))
    plt.plot(D_loss, label='discriminitive loss')
    plt.plot(GAN_loss, label='generative loss')
    plt.legend()
    plt.ylim((0,ylim))
    plt.savefig('output/GAN_plot.png')

def train( model_name,epochs = 10,batch_size = 128):
    path_train = "/home/sushobhan/git/GAN/"

    gt_data,gt_label = create_training_data()

    G = generator(nodes = 200, input_shape = 100)
    G.compile(loss='binary_crossentropy', optimizer=opt_gan)

    D = discriminator(input_shape = (1,28,28), dropout_rate = 0.5, filters = 256)
    D.compile(loss='binary_crossentropy', optimizer=opt_d)

    # D = load_model(path_train+'D_base.h5')

    GAN = Generative_adversary(D,G, shape = 100)
    GAN.compile(loss='binary_crossentropy', optimizer=opt_gan)


    # beg = time.time()

    # for i in range(2):

    #     for j in range(gt_data.shape[0]//batch_size):

    #         d_loss = train_D(D,G,(gt_data[j*batch_size:(j+1)*batch_size],
    #             gt_label[j*batch_size:(j+1)*batch_size]),
    #              batch_size = batch_size)
    #         print i,j, d_loss
    # D.save("D_base.h5")

    # print 'time: ', time.time()-beg

    D_loss = []
    GAN_loss =[]

    for i in range(epochs):

        print "Iteration :" + str(i)
        beg = time.time()

        for j in range(gt_data.shape[0]//batch_size):

            d_loss = train_D(D,G,[gt_data[j*batch_size:(j+1)*batch_size],
                gt_label[j*batch_size:(j+1)*batch_size]],
                 batch_size = batch_size)

            gan_loss = train_GAN(GAN,D, batch_size=batch_size)

            if j%10==0:
                print 'batch: ', str(j)
                print 'D loss: ', str(d_loss), ' GAN loss: ', str(gan_loss)

        print 'time: ', time.time()-beg
        D_loss.append(d_loss)
        GAN_loss.append(gan_loss)
        plot_loss(D_loss,GAN_loss,5.0)
        if not os.path.exists(path_train+'models/'+model_name+'/'):
            os.makedirs(path_train+'models/'+model_name+'/')

        G.save(path_train+'models/'+model_name+'/'+model_name+"G.h5")
        D.save(path_train+'models/'+model_name+'/'+model_name+"D.h5")

def generate(model_name, batch_size = 128):
    path_test = "/home/sushobhan/git/GAN/models/"
    home = "/home/sushobhan/git/GAN/"

    data = np.zeros((batch_size,100))
    for i in range(batch_size):
        data[i,:] = np.random.uniform(-1,1,100)

    # f = h5py.File(path_test+model_name+'.h5', 'r+')
    # del f['optimizer_weights']
    # f.close()
        
    # G = generator(nodes = 200, input_shape = 100)
    # G.compile(loss='binary_crossentropy', optimizer='SGD')
    G = load_model(path_test+model_name+'/'+model_name+ 'G.h5')
    y_output = G.predict(data)

    print y_output.shape

    for i in range(batch_size):
        print i,
        if not os.path.exists(home+'output/'+model_name+'/'):
            os.makedirs(home+'output/'+model_name+'/')
        
        cv2.imwrite(home+'output/'+model_name+'/'+str(i)+'.png',(y_output[i,0]*127.5 + 127.5)//1)



def main():
    args = get_args()
    if args.mode == 'train':
        train(model_name = args.model,batch_size = args.batch_size, epochs = args.epochs)
    elif args.mode == 'generate':
        generate(model_name = args.model,batch_size = args.batch_size)
    print "Done!" 

    


if __name__ == '__main__':
    main()
