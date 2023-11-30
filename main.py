import model
import torch
import torchvision
import numpy as np
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

device = 'cuda'

#Preparing data
train_path = 'traindata/'
test_path = 'testdata/'
train_dataset = ImageFolder(train_path,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                 torchvision.transforms.RandomResizedCrop(size=(224, 224), antialias=True)]))
test_dataset = ImageFolder(test_path, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                torchvision.transforms.RandomResizedCrop(size=(224, 224), antialias=True)]))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

#initializing model
fishnet = model.FishNet().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fishnet.parameters(), lr = 0.0001)

#training loop
E = 100
loss_hist = []
for e in range(E):
	total_loss = 0
	train_acc = 0
	fishnet.train()
	for x,y in tqdm(train_loader):
		x = x.to(device)
		y = y.to(device)
		optimizer.zero_grad()
		output = fishnet(x)
		loss = loss_fn(output,y)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
		train_acc += (np.sum(np.array(output.argmax(dim=1).tolist()) == np.array(y.tolist())))/23000
	total_loss /= len(train_loader)


	valid_loss = 0 
	valid_acc = 0
	with torch.inference_mode():
		for x,y in valid_loader:
			x = x.to(device)
			y = y.to(device)
			target = fishnet(x)
			loss = loss_fn()
			valid_loss += loss.item()
			valid_acc += (np.sum(np.array(target.argmax(dim=1).tolist()) == np.array(y.tolist())))/2000
	valid_loss /= len(test_loader)

	print(e , ": " , total_loss, ", ", train_acc,", ",valid_loss,", ", valid_acc)