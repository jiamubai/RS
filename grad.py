'''
Implement pretrained Resnet50 and use RS to finetune on Cifar10
'''

from learntosketch import *
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from pytorchtools import EarlyStopping

n_epochs = 150
batch_size_train = 1280
batch_size_test = 320
learning_rate = 0.01
lr = 1e-5
log_interval = 10
weight_decay = 0
times = 5000
dropout = 0.4
patience = 10

K = 7
R = 800
aggregation = 'linear'
backprop = 'STE'

device = 'cuda:0'
model = 'RS'

print('model: ', model)
print('n_epochs: ', n_epochs)
print('batch_size_train: ', batch_size_train)
print('batch_size_test: ', batch_size_test)
print('learning_rate: ', learning_rate)
print('lr: ', lr)
print('weight_decay: ', weight_decay)
print('times: ', times)
print('K: ', K)
print('R: ', R)
print('aggregation: ', aggregation)
print('backprop: ', backprop)
print('dropout: ', dropout)
print('Goal: Trying to address overfitting problem')
print('Change: Re-adjust learning rate')


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

## Load data

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
transform_train = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: F.pad(
    #     Variable(x.unsqueeze(0), requires_grad=False),
    #     (4, 4, 4, 4), mode='reflect').data.squeeze()),
    # transforms.ToPILImage(),
    # transforms.ColorJitter(brightness=0.0),
    # transforms.RandomCrop(32),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # normalize,
])

# data prep for test set
transform_test = transforms.Compose([
    transforms.ToTensor()])

train = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('~/torch_datasets', train=True, download=True,
                             transform=transform_train),batch_size=batch_size_train, shuffle=True)

# split train and validation dataset
train_subset, val_subset = torch.utils.data.random_split(train.dataset, (40000, 10000), generator=torch.Generator().manual_seed(random_seed))

train_loader = torch.utils.data.DataLoader(dataset=train_subset, shuffle=True, batch_size=batch_size_train)
valid_loader = torch.utils.data.DataLoader(dataset=val_subset, shuffle=False, batch_size=batch_size_test)


test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('~/torch_datasets', train=False, download=True,
                             transform=transform_test), batch_size=batch_size_test, shuffle=True)

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

def test():
  net.to(device)
  net.eval()
  test_loss = 0
  train_loss = 0
  valid_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in train_loader:
      data, target = data.to(device), target.to(device)
      # data = torch.reshape(data, (len(target), -1))
      output = net(data)
      train_loss += criterion(output, target)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
    print('\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      train_loss, correct, len(train_loader.dataset),
      100. * correct / len(train_loader.dataset)))
    correct = 0
    for data, target in valid_loader:
      data, target = data.to(device), target.to(device)
      # data = torch.reshape(data, (len(target), -1))
      output = net(data)
      valid_loss += criterion(output, target)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
    print('\nValid set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      train_loss, correct, len(valid_loader.dataset),
      100. * correct / len(valid_loader.dataset)))
    correct = 0
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
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

net = resnet50(weights=ResNet50_Weights.DEFAULT)
if model == 'RS':
  num_ftrs = net.fc.in_features
  net.fc = SketchNetwork(K=K, R=R, d=num_ftrs, OUT=10, aggregation=aggregation, hash_func='SRP',
                                      dropout_rate=dropout, scale=5.0, backprop=backprop)
elif model == 'MLP':
  num_ftrs = net.fc.in_features
  net.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                           nn.ReLU(),
                           # nn.Dropout(0.2),
                           nn.Linear(256, 32),
                           nn.ReLU(),
                           # nn.Dropout(0.2),
                           nn.Linear(32, 10))

optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss().to(device)

#test()
early_stopping = EarlyStopping(patience=patience, verbose=True)

train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []

for epoch in range(1, n_epochs + 1):
    train(epoch)
    net.eval()
    train_loss = np.average(train_losses)
    avg_train_losses.append(train_loss)
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = criterion(output, target)
        valid_losses.append(loss.item())

    valid_loss = np.average(valid_losses)
    avg_valid_losses.append(valid_loss)

    epoch_len = len(str(n_epochs))

    print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ' +
                 f'valid_loss: {valid_loss:.5f}')

    print(print_msg)
    train_losses = []
    valid_losses = []
    early_stopping(valid_loss, net)

    if early_stopping.early_stop:
      print("Early stopping")
      break

# newest model performance
test()
net.load_state_dict(torch.load('checkpoint.pt'))
# check point model performance
test()
