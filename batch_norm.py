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

def custom_batch_norm1d(input_tensor, weight, bias, eps):
    
    var, mean = torch.var_mean(input_tensor, dim=0, unbiased=False)
    print(input_tensor.shape, mean.shape)
    normed_tensor = (input_tensor - mean)/(torch.sqrt(var + eps))
    normed_tensor = normed_tensor*weight + bias
    return normed_tensor

# input_tensor = torch.Tensor([[0.0, 0, 1, 0, 2], [0, 1, 1, 0, 10]])
# batch_norm = nn.BatchNorm1d(input_tensor.shape[1], affine=False)

# # Проверка происходит автоматически вызовом следующего кода
# # (раскомментируйте для самостоятельной проверки,
# #  в коде для сдачи задания должно быть закомментировано):
# # import numpy as np
# all_correct = True
# for eps_power in range(10):
#     eps = np.power(10., -eps_power)
#     batch_norm.eps = eps
#     batch_norm_out = batch_norm(input_tensor)
#     custom_batch_norm_out = custom_batch_norm1d(input_tensor, 1,0,eps)

#     all_correct &= torch.allclose(batch_norm_out, custom_batch_norm_out)
#     all_correct &= batch_norm_out.shape == custom_batch_norm_out.shape
# print(all_correct)





# input_size = 7
# batch_size = 5
# input_tensor = torch.randn(batch_size, input_size, dtype=torch.float)
# eps = 1e-3
# batch_norm = nn.BatchNorm1d(input_size, eps=eps)
# batch_norm.bias.data = torch.randn(input_size, dtype=torch.float)
# batch_norm.weight.data = torch.randn(input_size, dtype=torch.float)
# batch_norm.momentum = 0.5

# custom_batch_norm1d = CustomBatchNorm1d(batch_norm.weight.data,
#                                         batch_norm.bias.data, eps, batch_norm.momentum)

# # Проверка происходит автоматически вызовом следующего кода
# # (раскомментируйте для самостоятельной проверки,
# #  в коде для сдачи задания должно быть закомментировано):
# all_correct = True

# for i in range(8):
#     torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
#     norm_output = batch_norm(torch_input)
#     custom_output = custom_batch_norm1d(torch_input)
#     all_correct &= torch.allclose(norm_output, custom_output, atol=1e-06) \
#         and norm_output.shape == custom_output.shape

# batch_norm.eval()
# custom_batch_norm1d.eval()

# for i in range(8):
#     torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
#     norm_output = batch_norm(torch_input)
#     custom_output = custom_batch_norm1d(torch_input)
#     all_correct &= torch.allclose(norm_output, custom_output, atol=1e-06) \
#         and norm_output.shape == custom_output.shape
# print(all_correct)

def custom_batch_norm2d(input_tensor, eps):
    vshape = (input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]*input_tensor.shape[3])   
    var, mean = torch.var_mean(input_tensor.view(vshape), dim=(0,2), unbiased=False)
    print(var, mean)
    var = torch.tensor([torch.var(input_tensor.view(vshape)[:, ch,:],unbiased=False) for ch in range(input_tensor.shape[1])])
    mean = torch.tensor([torch.mean(input_tensor.view(vshape)[:, ch,:]) for ch in range(input_tensor.shape[1])])
     
    # var1, mean1 = torch.var_mean(input_tensor.view(vshape)[:, 0,:],unbiased=False)
    # print(var1, mean1)
    normed_tensor = torch.zeros(vshape)
    for ch in range(input_tensor.shape[1]):
        normed_tensor[:, ch, :] = (input_tensor.view(vshape)[:, ch, :] - mean[ch])/(torch.sqrt(var[ch] + eps))
    return normed_tensor.view(input_tensor.shape)


#for 2D images
# input_channels = 3
# batch_size = 3
# height = 2
# width = 2
# eps = 1e-3
# batch_norm_2d = nn.BatchNorm2d(input_channels, affine=False, eps=eps)
# input_tensor = torch.randn(batch_size, input_channels, height, width, dtype=torch.float)
# norm_output = batch_norm_2d(input_tensor)
# custom_output = custom_batch_norm2d(input_tensor, eps)
# #print(norm_output)
# #print(custom_output)
# print(torch.allclose(norm_output, custom_output) and norm_output.shape == custom_output.shape)

eps = 1e-10
def custom_layer_norm(input_tensor, eps):
    vshape = (input_tensor.shape[0], np.prod(input_tensor.shape[1:]))   
    #vshape = (input_tensor.shape[0], -1)
    print(input_tensor.shape, vshape)
    var, mean = torch.var_mean(input_tensor.view(vshape), dim=1, unbiased=False)
    print(var, mean)
    # var = torch.var(input_tensor.view(vshape),dim=1,unbiased=False)
    # mean = torch.mean(input_tensor.view(vshape), dim=1)
     
    # var1, mean1 = torch.var_mean(input_tensor.view(vshape)[:, 0,:],unbiased=False)
    # print(var1, mean1)
    # normed_tensor = torch.zeros(vshape)
    # for ch in range(input_tensor.shape[1]):
    #     normed_tensor[:, ch, :] = (input_tensor.view(vshape)[:, ch, :] - mean[ch])/(torch.sqrt(var[ch] + eps))
    
    normed_tensor = (input_tensor.view(vshape).transpose(0,1) - mean)/(torch.sqrt(var + eps))   
    return normed_tensor.transpose(0,1).view(input_tensor.shape)

# all_correct = True
# for dim_count in range(3, 9):
#     input_tensor = torch.randn(*list(range(3, dim_count + 2)), dtype=torch.float)
#     layer_norm = nn.LayerNorm(input_tensor.size()[1:], elementwise_affine=False, eps=eps)

#     norm_output = layer_norm(input_tensor)
#     custom_output = custom_layer_norm(input_tensor, eps)

#     all_correct &= torch.allclose(norm_output, custom_output, 1e-2)
#     all_correct &= norm_output.shape == custom_output.shape
# print(all_correct)


# eps = 1e-3

# batch_size = 5
# input_channels = 2
# input_length = 30

# instance_norm = nn.InstanceNorm1d(input_channels, affine=False, eps=eps)

# input_tensor = torch.randn(batch_size, input_channels, input_length, dtype=torch.float)


def custom_instance_norm1d(input_tensor, eps):
    var = torch.var(input_tensor,dim=2,keepdim=True,unbiased=False)
    mean = torch.mean(input_tensor, keepdim=True, dim=2)
    normed_tensor = (input_tensor - mean)/(torch.sqrt(var + eps))   
    
    return normed_tensor

# norm_output = instance_norm(input_tensor)
# custom_output = custom_instance_norm1d(input_tensor, eps)
# print(torch.allclose(norm_output, custom_output, atol=1e-06) and norm_output.shape == custom_output.shape)

channel_count = 6
eps = 1e-3
batch_size = 20
input_size = 2

input_tensor = torch.randn(batch_size, channel_count, input_size)


def custom_group_norm(input_tensor, groups, eps):
   # vshape = (input_tensor.shape[0], input_tensor.shape[1]//groups, -1)
    vshape = input_tensor.shape
    input_tensor = input_tensor.reshape(input_tensor.shape[0], groups, -1)
    print(input_tensor.shape, vshape)
    var = torch.var(input_tensor,dim=2,keepdim=True,unbiased=False)
    mean = torch.mean(input_tensor, keepdim=True, dim=2)
    normed_tensor = (input_tensor - mean)/(torch.sqrt(var + eps))   

    return normed_tensor.reshape(vshape)

all_correct = True
for groups in [1, 2, 3, 6]:
    group_norm = nn.GroupNorm(groups, channel_count, eps=eps, affine=False)
    norm_output = group_norm(input_tensor)
    custom_output = custom_group_norm(input_tensor, groups, eps)
    all_correct &= torch.allclose(norm_output, custom_output, 1e-3)
    all_correct &= norm_output.shape == custom_output.shape
print(all_correct)