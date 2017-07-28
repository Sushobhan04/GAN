import torch
from torch.autograd import Variable
from torch.utils.serialization import load_lua
from torchvision import datasets
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 
import os
import sys
import time

class ConvRelu(nn.Module):
	def __init__(self,in_channels,out_channnels, kernel_size=3,stride=1,padding=1):
		super(ConvRelu,self).__init__()
		self.conv = nn.conv2d(in_channels,out_channnels,kernel_size=kernel_size,
			stride=stride, padding=padding, bias=False)
		self.relu = nn.ReLU()

	def forward(self,x):
		out = self.conv(x)
		out = self.relu(out)

		return out

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()

		self.conv1 = ConvRelu(1,64)
		self.conv2 = ConvRelu(64,128)
		self.conv3 = ConvRelu(128,256)
		self.conv4 = ConvRelu(256,64)
		self.conv5 = ConvRelu(64,1)

		self.linear1 = nn.ReLU()nn.Linear(28*28,1024)
		self.linear2 = nn.Linear(1024,1)

	def forward(self,x):

		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		out = F.relu(self.linear1(out))
		out = F.sigmoid(self.linear2(out))

		return out

class GAN(nn.Module):
	def __init__(self,lr=0.0001):
		super(GAN,self).__init__()

		self.discriminator = Discriminator().cuda()
		self.mse = nn.MSELoss().cuda()
		self.lr = lr
		self.optimizer = torch.optim.Adam(filter(lamda p: p.requires_grad,
			self.discriminator.parameters()),lr = self.lr)
		self.loss = None

	def loss(self,y_pred,y):
		self.loss = self.mse(y_pred,y)

	def load_data(self):
		train = datasets.MNIST('../data',train=True,download=True)
		# test = datasets.MNIST('../data',train=False,download=True)
		self.train_loader = torch.utils.data.DataLoader(train,batch_size=128, shuffle=True)

	def run_discriminator(self):
		
		y_pred = self.discriminator()

	def run(self,epochs):





		
