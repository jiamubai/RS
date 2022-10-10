'''
Implement pretrained Resnet50 and use RS to finetune final layer on Cifar10
'''

from learntosketch import *
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights
import torch
import torchvision

n_epochs = 50
batch_size_train = 256
batch_size_test = 1000
learning_rate = 0.02
lr = 1e-5
momentum = 0.9
log_interval = 10
weight_decay = 5e-4

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

## Load data
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('~/torch_datasets', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('~/torch_datasets', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  net.to(device)
  net.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = net(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      # torch.save(net.state_dict(), '~/results/model.pth')
      # torch.save(optimizer.state_dict(), '~/results/optimizer.pth')

def test():
  net.to(device)
  net.eval()
  test_loss = 0
  train_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      # data = torch.reshape(data, (len(target), -1))
      output = net(data)
      test_loss += criterion(output, target)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  # test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  # train_loss /= len(train_loader.dataset)
  # train_losses.append(train_losses)
  print('\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# net = SketchNetwork(K = 14, R = 6000, d = 28*28, OUT = 10, aggregation = 'avg', dropout_rate = 0.0, hash_func = 'P-stable')
# device = 'cuda:3'
device = input("Device: ")
model= input("Model(RS/MLP/Pretrain inference): ")
net = resnet50(weights=ResNet50_Weights.DEFAULT)
# net = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
if model == 'RS':
  for param in net.parameters():
    param.requires_grad = False
  num_ftrs = net.fc.in_features
  net.fc = SketchNetwork(K=4, R=4000, d=num_ftrs, OUT=10, aggregation='avg', hash_func='SRP',
                                      dropout_rate=0.0, scale=5.0, backprop='STE')
elif model == 'MLP':
  # for param in net.parameters():
  #   param.requires_grad = False
  # num_ftrs = net.fc.in_features
  net.fc = torch.nn.Linear(net.fc.in_features, 10)
  torch.nn.init.xavier_uniform_(net.fc.weight)
  # net.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
  #                          nn.ReLU(),
  #                          # nn.Dropout(0.2),
  #                          nn.Linear(256, 32),
  #                          nn.ReLU(),
  #                          # nn.Dropout(0.2),
  #                          nn.Linear(32, 10))

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
params_1x = [param for name, param in net.named_parameters() if 'fc' not in str(name)]
# lr = int(input("Learning rate:"))
# optimizer = torch.optim.Adam([{'params':params_1x}, {'params': net.fc.parameters(), 'lr': lr*1000}], lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss().to(device)
# net.to('cuda:3')
test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
