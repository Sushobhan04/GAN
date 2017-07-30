import os
import numpy as np 
import torch
from torchvision import datasets


class Dataset(object):
	def __init__(self):
		super(Dataset,self).__init__()
		self.train_loader = None

	def load_data(self):
		train = datasets.MNIST('../data',train=True,download=True)
		# test = datasets.MNIST('../data',train=False,download=True)
		self.train_loader = torch.utils.data.DataLoader(train,batch_size=config.batch_size, shuffle=True)
