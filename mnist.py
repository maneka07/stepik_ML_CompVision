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


X_train = MNIST_train.data.float()
X_test = MNIST_test.data.float()
ytrain = MNIST_train.targets
ytest = MNIST_test.targets
X_train = X_train.reshape([60000, 784])
X_test = X_test.reshape([X_test.shape[0], X_test.shape[1]*X_test.shape[2]])

nclasses = len(MNIST_train.classes)
nneurons = 100
nfeat = X_train.shape[1]
#num_net = MNISTNetFC(nfeat, nclasses, nneurons)

torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#num_net = num_net.to(device)

loss = torch.nn.CrossEntropyLoss()
lr = 0.001
#optAdam = torch.optim.Adam(num_net.parameters(), lr=lr)

#Will test different optimizers
optAdam = torch.optim.Adam
optSGD = torch.optim.SGD #(num_net.parameters(), lr=lr)
optRprop = torch.optim.Rprop #(num_net.parameters(), lr=lr)
optRMSprop = torch.optim.RMSprop #(num_net.parameters(), lr=lr)
opts = (optAdam, optSGD, optRprop, optRMSprop)

batch_sz = 100
X_test = X_test.to(device)
ytest = ytest.to(device)

nepochs = 40
nopts = len(opts)
train_acc_hist = np.zeros((nopts,nepochs))
train_loss_hist = np.zeros((nopts,nepochs))
test_acc_hist, test_loss_hist = np.zeros((nopts,nepochs)), np.zeros((nopts,nepochs))
avg_epoch_time = [0.0]*nopts

for i,opttype in enumerate(opts):
  num_net = MNISTNetFC(nfeat, nclasses, nneurons)
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
    test_loss_hist[i][epoch] = loss(pred, ytest)
    avg_epoch_time[i] += toc-tic
  
  avg_epoch_time[i] /= nepochs
  #print("avg epoch time {0:.4f}".format(avg_epoch_time[i]/float(nepochs)))

#!nvidia-smi
#torch.cuda.is_available()

for i, opt in enumerate(opts):
  print("for opt {0} accuracy {1:.4f} epoch time {2:.4f}".format(opt, test_acc_hist[i].mean(), avg_epoch_time[i]))

test_acc_hist[0]

plt.plot(train_acc_hist[0], label="Adam") 
plt.plot(test_acc_hist[1], label="SGD")
plt.plot(test_acc_hist[2], label="Rprop")
plt.plot(test_acc_hist[3], label="RMSprop")
plt.legend()

plt.plot(train_loss_hist, label="Train loss") 
plt.plot(test_loss_hist, label="Test loss")
plt.legend()

#test_acc_hist[0]

