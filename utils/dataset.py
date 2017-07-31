import os
import numpy as np 
import torch
from torchvision import datasets


class Dataset(object):
	def __init__(self,dataset):
		super(Dataset,self).__init__()
		self.dataset = dataset

	def batch_generator(self,batch_size):
		while 1:
			batch = []
			for i in xrange(batch_size):
				batch.append(np.array(self.dataset[i][0]))

			batch = np.array(batch)
			yield batch

