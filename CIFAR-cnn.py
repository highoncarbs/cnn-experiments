# CNN on  CIFAR10 Dataset
# Each input image is 3x32x32

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as trans 
import matplotlib.pyplot as plt
import numpy as np 

# Composing the images

transform = trans.Compose(
		[	trans.ToTensor(),
			trans.Normalize((0.5 , 0.5 , 0.5) ,(0.5 , 0.5 , 0.5)) 
		])
trainset = torchvision.datasets.CIFAR10(root='../data' , train=True ,download =True , transform=transform)

trainloader = torch.utils.data.DataLoader(trainset , batch_size = 4 , shuffle =True , num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data' , train=False , download=True , transform=transform)

testloader = torch.utils.data.DataLoader(testset , batch_size=4 , shuffle=False , num_workers=2)

classes = ('plane' , 'car' , 'bird' , 'cat' , 'deer' , 'dog' , 'frog' , 'horse' , 'ship' , 'truck')

# Visualize images 

def imshow(img):
	img = img/2 + 0.5
	nimg = img.numpy()
	plt.imshow(np.transpose(nimg , (1,2,0)))
