import torch
import torch.nn as nn
import torchvision.datasets as dataset 
import torchvision.transforms as tranforms
from torch.autograd import Variable

# Parameters

n_epoch = 5
learning_rate = 1e-3
batch_size = 100

train_dataset = dataset.MNIST(root = '../data/ ', 
						train=True , 
						transform=transform.ToTensor() , 
						download=True)

test_dataset = dataset.MNIST(root='../data/',
						train=False,
						transform = tranforms.ToTensor())

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
					nn.MaxPool2d(2);
					)
		self.fc = nn.Linear(7*7*32 , 10)

	def forward(self , x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(out.size(0) , -1)
		out = self.fc(out)

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