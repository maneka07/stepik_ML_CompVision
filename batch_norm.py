# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:21:28 2022
Implementation of batch-normalization
@author: Tatiana
"""
import numpy as np
import torch
import torch.nn as nn

def custom_batch_norm1d(input_tensor, eps):
    normed_tensor = # Напишите в этом месте нормирование входного тензора
    return normed_tensor


input_tensor = torch.Tensor([[0.0, 0, 1, 0, 2], [0, 1, 1, 0, 10]])
batch_norm = nn.BatchNorm1d(input_tensor.shape[1], affine=False)

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# import numpy as np
# all_correct = True
# for eps_power in range(10):
#     eps = np.power(10., -eps_power)
#     batch_norm.eps = eps
#     batch_norm_out = batch_norm(input_tensor)
#     custom_batch_norm_out = custom_batch_norm1d(input_tensor, eps)

#     all_correct &= torch.allclose(batch_norm_out, custom_batch_norm_out)
#     all_correct &= batch_norm_out.shape == custom_batch_norm_out.shape
# print(all_correct)
