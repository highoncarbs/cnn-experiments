import torch
import torch.nn as nn
import torchvision.datasets as dataset 
import torchvision.transforms as transforms
from torch.autograd import Variable

# Parameters

n_epoch = 5
learning_rate = 1e-3
batch_size = 100

train_dataset = dataset.MNIST(root = '../data/mnist', 
						train=True , 
						transform=transforms.ToTensor() , 
						download=True)

test_dataset = dataset.MNIST(root='../data/mnist',
						train=False,
						transform = transforms.ToTensor())

# Loading Data

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
								batch_size = batch_size,
								shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
								batch_size = batch_size,
								shuffle = False)

class CNN(nn.Module):
	def __init__(self):
		super(CNN , self).__init__()
		self.layer1 = nn.Sequential(
					# Param( in channel , out channel , kern_size, stride , padding)
					# for square kernel , input -> <value>
					# for custom kernel , input -> <width , height> of kernel
					nn.Conv2d(1 , 16 , kernel_size = 5 , padding =2),
					nn.BatchNorm2d(16),
					nn.ReLU(),
					nn.MaxPool2d(2))
		self.layer2 = nn.Sequential(
					nn.Conv2d(16 , 32, kernel_size=5 , padding=2),
					nn.BatchNorm2d(32),
					nn.ReLU(),
					nn.MaxPool2d(2)
					)
		self.fc = nn.Linear(7*7*32 , 10)

	def forward(self , x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(out.size(0) , -1)
		out = self.fc(out)
		return out

cnn = CNN()

# Loss Function

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters() , lr = learning_rate)

# Training
for epoch in range(n_epoch):
	for i , (images, labels) in enumerate(train_loader):
		images = Variable(images)
		labels = Variable(labels)

		# Running Forward , backward an optimization

		optimizer.zero_grad() # Setting the gradients 0
		output = cnn(images)
		loss = criterion(output , labels)
		loss.backward()
		optimizer.step()

		if(i+1)% 100 == 0:
			print ('Epoch [%d%d] , iter [%d%d] Loss: %.4f' 
					%(epoch+1 , n_epoch , i+1 , len(train_dataset)//batch_size , loss.data[0]))

# Running the model
# eval() runs the python code within itself

cnn.eval()
total = 0
correct = 0

for images, labels in test_loader:
	images = Variable(images)
	output = cnn(images)
	_ , pred = torch.max(output.data , 1)
	total += labels.size(0)
	correct += (pred == labels).sum()

print(' Accuracy on 10k test images : %d %%' % (100*correct/total))

torch.save(cnn.state_dict() , 'cnn.pkl')		

'''
Output

Epoch [15] , iter [100600] Loss: 0.2221
Epoch [15] , iter [200600] Loss: 0.0690
Epoch [15] , iter [300600] Loss: 0.0889
Epoch [15] , iter [400600] Loss: 0.0245
Epoch [15] , iter [500600] Loss: 0.0795
Epoch [15] , iter [600600] Loss: 0.0713
Epoch [25] , iter [100600] Loss: 0.1015
Epoch [25] , iter [200600] Loss: 0.0209
Epoch [25] , iter [300600] Loss: 0.0882
Epoch [25] , iter [400600] Loss: 0.0117
Epoch [25] , iter [500600] Loss: 0.0419
Epoch [25] , iter [600600] Loss: 0.0262
Epoch [35] , iter [100600] Loss: 0.1080
Epoch [35] , iter [200600] Loss: 0.0663
Epoch [35] , iter [300600] Loss: 0.0538
Epoch [35] , iter [400600] Loss: 0.1063
Epoch [35] , iter [500600] Loss: 0.0250
Epoch [35] , iter [600600] Loss: 0.0112
Epoch [45] , iter [100600] Loss: 0.0476
Epoch [45] , iter [200600] Loss: 0.0493
Epoch [45] , iter [300600] Loss: 0.0659
Epoch [45] , iter [400600] Loss: 0.0148
Epoch [45] , iter [500600] Loss: 0.0086
Epoch [45] , iter [600600] Loss: 0.0125
Epoch [55] , iter [100600] Loss: 0.0086
Epoch [55] , iter [200600] Loss: 0.0220
Epoch [55] , iter [300600] Loss: 0.0461
Epoch [55] , iter [400600] Loss: 0.0039
Epoch [55] , iter [500600] Loss: 0.0331
Epoch [55] , iter [600600] Loss: 0.0238
Accuracy on 10k test images : 99 %


'''