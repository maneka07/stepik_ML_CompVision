# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:21:28 2022
Implementation of batch-normalization
@author: Tatiana
"""
import numpy as np
import torch
import torch.nn as nn


class CustomBatchNorm1d:
    def __init__(self, weight, bias, eps, momentum):
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.momentum = momentum
        self.train_mode = True
        self.EMA_mean = torch.zeros(weight.shape[0])
        self.EMA_var = torch.ones(weight.shape[0])

    def __call__(self, input_tensor):
        if self.train_mode:
            var, mean = torch.var_mean(input_tensor, dim=0, unbiased=False)
            normed_tensor = (input_tensor - mean)/(torch.sqrt(var + self.eps))
            #update moving averages
            n = input_tensor.shape[0]
            self.EMA_mean = self.momentum*self.EMA_mean + (1-self.momentum)*mean
            self.EMA_var = self.momentum*self.EMA_var + (1-self.momentum)*n*var/(n-1)
        else:
            normed_tensor = (input_tensor - self.EMA_mean)/torch.sqrt(self.EMA_var + self.eps)
        normed_tensor = normed_tensor*self.weight + self.bias
        return normed_tensor
    
    def eval(self):
        self.train_mode = False

# def custom_batch_norm1d(input_tensor, weight, bias, eps):
    
#     var, mean = torch.var_mean(input_tensor, dim=0, unbiased=False)
#     normed_tensor = (input_tensor - mean)/(torch.sqrt(var + eps))
#     normed_tensor = normed_tensor*weight + bias
#     return normed_tensor


input_size = 7
batch_size = 5
input_tensor = torch.randn(batch_size, input_size, dtype=torch.float)
eps = 1e-3
batch_norm = nn.BatchNorm1d(input_size, eps=eps)
batch_norm.bias.data = torch.randn(input_size, dtype=torch.float)
batch_norm.weight.data = torch.randn(input_size, dtype=torch.float)
batch_norm.momentum = 0.5

custom_batch_norm1d = CustomBatchNorm1d(batch_norm.weight.data,
                                        batch_norm.bias.data, eps, batch_norm.momentum)

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
all_correct = True

for i in range(8):
    torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
    norm_output = batch_norm(torch_input)
    custom_output = custom_batch_norm1d(torch_input)
    all_correct &= torch.allclose(norm_output, custom_output, atol=1e-06) \
        and norm_output.shape == custom_output.shape

batch_norm.eval()
custom_batch_norm1d.eval()

for i in range(8):
    torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
    norm_output = batch_norm(torch_input)
    custom_output = custom_batch_norm1d(torch_input)
    all_correct &= torch.allclose(norm_output, custom_output, atol=1e-06) \
        and norm_output.shape == custom_output.shape
print(all_correct)