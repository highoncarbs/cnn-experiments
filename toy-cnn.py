import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as fun 
import sys
import torch.optim as optim 

class Net(nn.Module):

	def __init__(self):
		super(Net , self).__init__()

		# Kernel size = 5
		self.conv1 = nn.Conv2d(1,6,5)
		self.conv2 = nn.Conv2d(6,16,5)

		# Affine connections

		self.lin1 = nn.Linear(16*5*5 , 120)
		self.lin2 = nn.Linear(120 , 84)
		self.lin3 = nn.Linear(84 , 10)
	
	def forward(self , x):

		# Max pooling = 2
		x = fun.max_pool2d(fun.relu(self.conv1(x)) , 2)
		x = fun.max_pool2d(fun.relu(self.conv2(x)) , 2)		
		x = x.view(-1 , self.num_flat(x))
		x = fun.relu(self.lin1(x))
		x = fun.relu(self.lin2(x))
		x = self.lin3(x)
		return x

	def num_flat(self,x):
		size = x.size()[1:]
		num_feat = 1
		for i in size:
			num_feat *= i
		return num_feat

net = Net()

input = Variable(torch.randn(1, 1, 32, 32))
out	= net(input)
target = Variable(torch.randn(1,10)) # Dummy target	

criterion = nn.MSELoss()
loss = criterion(out , target)
print(loss)
print(loss.creator)

net.zero_grad() # Zeros the buffer gradients
print(' conv1 grad before backward')
print(net.conv2.bias.grad)

loss.backward()

print(' conv 1 grad after backward')
print(loss)
print(net.conv1.bias.grad)

# Updating the weights

# lr = 0.01
# for param in net.parameters():
# 	param.data.sub_(param.grad.data*lr)

# Method 2 of updating the wts with SGD 

optimizer = optim.SGD(net.parameters() , lr =0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output , target)
loss.backward() 
optimizer.step() # Automatic updating weights