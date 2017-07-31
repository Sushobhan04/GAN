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
import config
from utils.mnistGAN import GAN

def main(args):
	gan = GAN()
	gan.load_data()

	loader = gan.dataset

	print loader[0][0]

	for i,data in enumerate(loader):
		print np.array(data[1])
		print i
		break
	

if __name__=='__main__':
	main(sys.argv)