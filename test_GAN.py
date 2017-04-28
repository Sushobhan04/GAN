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

def crop(set,N):
    h = set.shape[2]
    w = set.shape[3]

    return set[:,:,N:h-N,N:w-N]


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

    path_test = "/home/sushobhan/git/GAN/"
    home = "/home/sushobhan/git/GAN/"
    model_name = sys.argv[1]

    trails = int(sys.argv[2])
    data = np.random.random((trails,100))
        
    model = load_model(path_test+model_name+'.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam')
    y_output = model.predict(data)

    print y_output.shape

    for i in range(trails):
        print i
        cv2.imwrite(home+'output/'+str(i)+'.png',(y_output[i,0]*255)//1)


if __name__ == '__main__':
    main()
