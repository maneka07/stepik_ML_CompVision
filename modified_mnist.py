# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:19:24 2021

@author: Tatiana
"""

import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt

random.seed(25)
np.random.seed(25)
torch.manual_seed(25)
torch.cuda.manual_seed(25)
torch.backends.cudnn.deterministic = True

import torchvision.datasets
MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)

#MNIST task using a fully connected NN
class MNISTNetFC(torch.nn.Module):
  def __init__(self, nfeat, nclasses, nneurons):
    super(MNISTNetFC, self).__init__()
    self.fc1 = torch.nn.Linear(nfeat, nneurons)
    self.act1 = torch.nn.Sigmoid()
    self.fc2 = torch.nn.Linear(nneurons, nclasses)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act1(x)
    x = self.fc2(x)
    return x
  
  def predict(self, x):
    sm = torch.nn.Softmax(dim=1)
    x = self.forward(x)
    return sm(x)


class MNISTLeNet5(torch.nn.Module):
    def __init__(self):
        super(MNISTLeNet5, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.act1  = torch.nn.Tanh()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
       
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.act2  = torch.nn.Tanh()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc1   = torch.nn.Linear(5 * 5 * 16, 120)
        self.act3  = torch.nn.Tanh()
        
        self.fc2   = torch.nn.Linear(120, 84)
        self.act4  = torch.nn.Tanh()
        
        self.fc3   = torch.nn.Linear(84, 10)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)
        
        return x


X_train = MNIST_train.data.float()
X_test = MNIST_test.data.float()
ytrain = MNIST_train.targets
ytest = MNIST_test.targets

nclasses = len(MNIST_train.classes)
nneurons = 50
nfeat = X_train.shape[1]
#num_net = MNISTNetFC(nfeat, nclasses, nneurons)

torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

loss = torch.nn.CrossEntropyLoss()
lr = 0.001
#optAdam = torch.optim.Adam(num_net.parameters(), lr=lr)


#Will test different optimizers
optAdam = torch.optim.Adam
optSGD = torch.optim.SGD #(num_net.parameters(), lr=lr)
optRprop = torch.optim.Rprop #(num_net.parameters(), lr=lr)
optRMSprop = torch.optim.RMSprop #(num_net.parameters(), lr=lr)
#opts = (optAdam, optSGD, optRprop, optRMSprop)
opts = (optAdam,)

batch_sz = 100
nopts = len(opts)
nepochs = 40
train_acc_hist = np.zeros((nopts,nepochs))
train_loss_hist = np.zeros((nopts,nepochs))
test_acc_hist, test_loss_hist = np.zeros((nopts,nepochs)), np.zeros((nopts,nepochs))
avg_epoch_time = [0.0]*nopts

testNN = MNISTLeNet5
if testNN == MNISTNetFC:
    #flatten the data from 2D into 1D vectors since FC networks can't work
    #with multi-dimensional data
    X_train = X_train.reshape([60000, 784])
    X_test = X_test.reshape([X_test.shape[0], X_test.shape[1]*X_test.shape[2]])
elif testNN == MNISTLeNet5:
    #add dimension for input channels, images are bw -> only one channel
    X_train = X_train.unsqueeze(1).float()
    X_test = X_test.unsqueeze(1).float()

X_test = X_test.to(device)
ytest = ytest.to(device)

for i,opttype in enumerate(opts):
    if testNN == MNISTNetFC:
        num_net = MNISTNetFC(nfeat, nclasses, nneurons)
    elif testNN == MNISTLeNet5:
        num_net = MNISTLeNet5()
        
    num_net.to(device)
    opt = opttype(num_net.parameters(), lr=lr)
    for epoch in range(nepochs):
      tic = time.perf_counter()
      order = np.random.permutation(range(X_train.shape[0]))
      for batch_ind in range(0, X_train.shape[0], batch_sz):
        opt.zero_grad()
  
        xbatch = X_train[order[batch_ind:batch_ind+batch_sz]].to(device)
        ybatch = ytrain[order[batch_ind:batch_ind+batch_sz]].to(device)
        
        pred = num_net.forward(xbatch)
        acc = (ybatch==pred.argmax(dim=1)).float().mean()
        train_acc_hist[i][epoch] += acc
        loss_val = loss(pred, ybatch)
        train_loss_hist[i][epoch] += loss_val
        loss_val.backward()
        opt.step()
      train_acc_hist[i][epoch] = train_acc_hist[i][epoch]/float(len(range(0, X_train.shape[0], batch_sz)))
      train_loss_hist[i][epoch] = train_loss_hist[i][epoch]/float(len(range(0, X_train.shape[0], batch_sz)))
      #if epoch % 200 == 0:
      pred = num_net.forward(X_test)
      toc = time.perf_counter()
      acc = (ytest == pred.argmax(dim=1)).float().mean()
      test_acc_hist[i][epoch] = acc
      test_loss_hist[i][epoch] = loss(pred, ytest).data.cpu()
      avg_epoch_time[i] += toc-tic
    
    avg_epoch_time[i] /= nepochs
  #print("avg epoch time {0:.4f}".format(avg_epoch_time[i]/float(nepochs)))

#!nvidia-smi
#torch.cuda.is_available()

for i, opt in enumerate(opts):
  print("for opt {0} accuracy {1:.4f} epoch time {2:.4f}".format(opt, test_acc_hist[i].mean(), avg_epoch_time[i]))

test_acc_hist[0]

plt.plot(train_acc_hist[0], label="Adam") 
# plt.plot(test_acc_hist[1], label="SGD")
# plt.plot(test_acc_hist[2], label="Rprop")
# plt.plot(test_acc_hist[3], label="RMSprop")
plt.legend()

plt.plot(train_loss_hist, label="Train loss") 
plt.plot(test_loss_hist, label="Test loss")
plt.legend()

#test_acc_hist[0]

