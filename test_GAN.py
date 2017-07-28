from keras.models import load_model
import h5py
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
import sys
from skimage.measure import compare_psnr
import cv2
import random
import tensorflow as tf
from keras.optimizers import SGD, Adadelta, Adam
import os


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

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# opt_gan = SGD(lr=0.0005, momentum=0.9, nesterov=True)

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



def normalize(arr):
    ma = np.max(arr,axis=(2,3))
    mi = np.min(arr,axis=(2,3))
    ma = np.expand_dims(np.expand_dims(ma,axis=2),axis=3)
    mi = np.expand_dims(np.expand_dims(mi,axis=2),axis=3)

    arr = np.divide((arr - mi),(ma-mi))

    return arr

def BatchGenerator(files,batch_size,dtype = 'train', N=0):
    while 1:
        for file in files:
            curr_data = h5py.File(file,'r')
            data = np.array(curr_data[dtype]['data'][()])
            label = np.array(curr_data[dtype]['label'][()])
            # print data.shape, label.shape

            for i in range((data.shape[0]-1)//batch_size + 1):
                # print 'batch: '+ str(i)
                data_bat = data[i*batch_size:(i+1)*batch_size,]
                label_bat = label[i*batch_size:(i+1)*batch_size,]
                yield (data_bat, crop(label_bat,N))

def main():

    path_test = "/home/sushobhan/git/GAN/models/"
    home = "/home/sushobhan/git/GAN/"
    model_name = sys.argv[1]

    trails = int(sys.argv[2])
    data = np.zeros((trails,100))
    for i in range(trails):
        data[i,:] = np.random.uniform(-1,1,100)

    # f = h5py.File(path_test+model_name+'.h5', 'r+')
    # del f['optimizer_weights']
    # f.close()
        
    G = generator(nodes = 200, input_shape = 100)
    G.compile(loss='binary_crossentropy', optimizer='SGD')
    G.load_weights(path_test+model_name+ 'G.weights')
    y_output = G.predict(data)

    print y_output.shape

    for i in range(trails):
        print i,
        if not os.path.exists(home+'output/'+model_name+'/'):
            os.makedirs(home+'output/'+model_name+'/')
        
        cv2.imwrite(home+'output/'+model_name+'/'+str(i)+'.png',(y_output[i,0]*127.5 + 127.5)//1)


if __name__ == '__main__':
    main()
