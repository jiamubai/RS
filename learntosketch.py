import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets
from hashes import *

class SketchNetwork(nn.Module):
    def __init__(self, K, R, d, OUT, aggregation, dropout_rate, hash_func, backprop=None, scale = 1.0):
        super(SketchNetwork, self).__init__()
        self.K = K
        self.R = R
        self.OUT = OUT
        self.d = d
        self.num_cel = 2**K
        self.scale = scale
        self.hash_func = hash_func
        self.bp = backprop
        if self.hash_func == 'SRP':
            self.h = SRP(self.K, self.R, self.d, self.OUT)
        elif self.hash_func == 'P-stable':
            self.h = PstableHash(self.R, self.d, self.OUT, self.scale)
        self.aggregation = aggregation
        # self.ste = StraightThroughEstimator()
        self.sketch = nn.Parameter(torch.Tensor(OUT, R, self.num_cel),requires_grad = True)
        nn.init.normal_( self.sketch, 0.0, 0.0001 )
        self.dropout_rate = dropout_rate
        self.dropout= None
        if(dropout_rate > 0.0):
            self.dropout = nn.Dropout(p = self.dropout_rate)
            
        if(self.aggregation == "linear"):
            self.agg = nn.Parameter( torch.Tensor(R), requires_grad = True)
            self.agg.data.fill_(1.0)

    def init_param(self, race, hashes):
        self.h.data = hashes
        self.sketch.data = race

    def gate(self):
        inner = torch.mul(self.sketch, self.sketch)
        softmax = torch.nn.Softmax(dim=0)
        g_prob = softmax(inner)
        out_sketch = torch.mul(inner, g_prob)
        # print('out sketch', out_sketch.shape)
        return out_sketch

    def forward(self, X):
        #
        with torch.no_grad():
            # X = torch.reshape(X, (X.shape[0], 28 * 28))
            hashcode = self.h.hash(X)
        if self.bp == 'STE':
            hashcode = STEFunction.apply(hashcode).long()
            out_sketch = self.sketch

        # hashcode = self.h.hash(X)
        else:
            out_sketch = self.gate()
        if self.hash_func == 'SRP':
            if( self.dropout != None):
                input = self.dropout(out_sketch) * (1.0 - self.dropout_rate)
                alphas = torch.gather( input = input , dim = 2, index = hashcode)
            else:
                alphas = torch.gather( input = out_sketch , dim = 2, index = hashcode)
        elif self.hash_func == 'P-stable':
            hashcode[hashcode < 0] = 0
            if( self.dropout != None):
                input = self.dropout(out_sketch) * (1.0 - self.dropout_rate)
                alphas = torch.gather( input = input , dim = 2, index = hashcode)
            else:
                alphas = torch.gather( input = out_sketch , dim = 2, index = hashcode)
        alphas = alphas.permute(2,0,1) # alphas [OUT, R, B] -> [B, OUT, R]

        if(self.aggregation == "avg"):
            predict = torch.mean(alphas, dim = 2)
        elif(self.aggregation == "linear"):
            alphas = alphas * self.agg
            predict = torch.mean(alphas, dim = 2)
        return predict

    
    def get_memory(self):
        agg_cost = 0
        if(self.aggregation == "linear"):
            agg_cost = self.OUT * self.R
        return self.h.get_memory() + (self.OUT * self.R * self.num_cel + agg_cost) * 32
    
    def get_flops(self):
        return self.h.get_flops() + self.R * self.OUT

'''

data = np.array([[1,1,2,1,1],[2,2,2,2,3],[3,1,3,3,2],[1,1,1,1,2],[3,3,3,3,2],[4,1,1,1,1]])
y = np.array([[1,2,3,1,3,1]])

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('~/torch_datasets', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               # torchvision.transforms.Normalize(
                               #   (0.1307,), (0.3081,))
                             ])),
  batch_size=256, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('~/torch_datasets', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               # torchvision.transforms.Normalize(
                               #   (0.1307,), (0.3081,))
                             ])),
  batch_size=1000, shuffle=True)

def test():
  net.eval()
  test_loss = 0
  train_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data = torch.reshape(data, (len(target), -1))
      output = net(data)
      test_loss += criterion(output, target)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  # test_loss /= len(test_loader.dataset)
  # test_losses.append(test_loss)
  # train_loss /= len(train_loader.dataset)
  # train_losses.append(train_losses)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

net = SketchNetwork(K=12, R=1200, d=28*28, OUT=10, aggregation='avg', hash_func='SRP',
                                      dropout_rate=0.0, scale=5.0, backprop=None)

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
for i in range(10):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print("Training batch: ", batch_idx)
        data = torch.reshape(data, (len(target), -1))
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            i, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))

    test()
'''
