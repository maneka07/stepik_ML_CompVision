# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:14:55 2021

@author: Tatiana
"""

import torch
from abc import ABC, abstractmethod
# Создаем входной массив из двух изображений RGB 3*3
input_images = torch.tensor(
      [[[[0,  1,  2],
         [3,  4,  5],
         [6,  7,  8]],

        [[9, 10, 11],
         [12, 13, 14],
         [15, 16, 17]],

        [[18, 19, 20],
         [21, 22, 23],
         [24, 25, 26]]],


       [[[27, 28, 29],
         [30, 31, 32],
         [33, 34, 35]],

        [[36, 37, 38],
         [39, 40, 41],
         [42, 43, 44]],

        [[45, 46, 47],
         [48, 49, 50],
         [51, 52, 53]]]])

correct_padded_images = torch.tensor(
       [[[[0.,  0.,  0.,  0.,  0.],
          [0.,  0.,  1.,  2.,  0.],
          [0.,  3.,  4.,  5.,  0.],
          [0.,  6.,  7.,  8.,  0.],
          [0.,  0.,  0.,  0.,  0.]],

         [[0.,  0.,  0.,  0.,  0.],
          [0.,  9., 10., 11.,  0.],
          [0., 12., 13., 14.,  0.],
          [0., 15., 16., 17.,  0.],
          [0.,  0.,  0.,  0.,  0.]],

         [[0.,  0.,  0.,  0.,  0.],
          [0., 18., 19., 20.,  0.],
          [0., 21., 22., 23.,  0.],
          [0., 24., 25., 26.,  0.],
          [0.,  0.,  0.,  0.,  0.]]],


        [[[0.,  0.,  0.,  0.,  0.],
          [0., 27., 28., 29.,  0.],
          [0., 30., 31., 32.,  0.],
          [0., 33., 34., 35.,  0.],
          [0.,  0.,  0.,  0.,  0.]],

         [[0.,  0.,  0.,  0.,  0.],
          [0., 36., 37., 38.,  0.],
          [0., 39., 40., 41.,  0.],
          [0., 42., 43., 44.,  0.],
          [0.,  0.,  0.,  0.,  0.]],

         [[0.,  0.,  0.,  0.,  0.],
          [0., 45., 46., 47.,  0.],
          [0., 48., 49., 50.,  0.],
          [0., 51., 52., 53.,  0.],
          [0.,  0.,  0.,  0.,  0.]]]])

#Add zero padding to images
def get_padding2d(input_images):
    nimgs = input_images.shape[0]
    nchans = input_images.shape[1]
    nrows = input_images.shape[2]
    ncols = input_images.shape[3]
    padded_images = torch.zeros((nimgs, nchans, nrows+2, ncols+2))
    print(padded_images.shape)
    for i in range(input_images.shape[0]):
        padded_images[i, 0: ,1:nrows+1,1:ncols+1] = input_images[i]         
    return padded_images

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
#print(torch.allclose(get_padding2d(input_images), correct_padded_images))

#Calculated the shape of the input tensor after a convolution layer has been
#applied
def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    nrows_in = input_matrix_shape[2]
    ncols_in = input_matrix_shape[3]
    nrows_out = (nrows_in+padding*2 - kernel_size)//stride+1
    ncols_out = (ncols_in+padding*2- kernel_size)//stride+1
    out_shape = [input_matrix_shape[0], out_channels, nrows_out, ncols_out]
    return out_shape

out = calc_out_shape(input_matrix_shape=[2, 3, 10, 10],
                   out_channels=10,
                   kernel_size=3,
                   stride=1,
                   padding=0)
#print(np.array_equal(
    # out,
    # [2, 10, 8, 8]))

# ... и ещё несколько подобных кейсов


# Abstract class for the convolution layer
class ABCConv2d(ABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def set_kernel(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, input_tensor):
        pass


# Wrapper class for torch.nn.Conv2d to unify the interface
class Conv2d(ABCConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding=0, bias=False)

    def set_kernel(self, kernel):
        self.conv2d.weight.data = kernel

    def __call__(self, input_tensor):
        return self.conv2d(input_tensor)

class myConv2dLoop(ABCConv2d):
    def __call__(self, input_tensor):
        if input_tensor == None:
            return None
        nimages, out_channels, nrows_out, ncols_out = calc_out_shape(
            input_tensor.shape, 
            self.out_channels, 
            self.kernel_size, 
            self.stride, 0)
       
        output_tensor = torch.zeros(input_tensor.shape[0]*self.out_channels*\
                                    nrows_out*ncols_out).reshape(\
                                    input_tensor.shape[0], self.out_channels, 
                                    nrows_out, ncols_out)
        for img in range(nimages):
            for out_ch in range(out_channels):
                fltr = self.kernel[out_ch]
                for in_ch in range(self.in_channels):
                    for row in range(0, nrows_out):
                        for col in range(0, ncols_out):
                            tmp = input_tensor[img][in_ch][row*self.stride:row*self.stride+self.kernel_size,col*self.stride:col*self.stride+self.kernel_size]
                            output_tensor[img][out_ch][row][col] += torch.mul(tmp, fltr[in_ch]).sum()       
        return output_tensor

class myConv2dMatmul(ABCConv2d):
    def __call__(self, input_tensor):
        if input_tensor == None:
            return None
        nimages, out_channels, nrows_out, ncols_out = calc_out_shape(
            input_tensor.shape, 
            self.out_channels, 
            self.kernel_size, 
            self.stride, 0)
        #print(f"input tensor {input_tensor[0]}")
        # output_tensor = torch.zeros(input_tensor.shape[0]*self.out_channels*\
        #                             nrows_out*ncols_out).reshape(\
        #                             input_tensor.shape[0], self.out_channels, 
        #                             nrows_out, ncols_out)
        img_height = input_tensor[0][0].shape[0]
        img_width = input_tensor[0][0].shape[1]
        #transform all images
       # input_tensor = input_tensor.reshape(nimages, input_tensor.numel()//nimages).transpose(0,1)
        #transform kernel into a matrix for matrix multiplication
        #print(self.kernel)
        km_cols = img_height*img_width*self.in_channels
        km_rows = self.out_channels*nrows_out*ncols_out
        kernel_mat = torch.zeros(km_rows,km_cols)
        for out_ch in range(self.out_channels):
            for in_ch in range(self.in_channels):
                fltr = self.kernel[out_ch][in_ch]
                in_ch_shift = in_ch*img_height*img_width
                out_ch_shift = out_ch*nrows_out*ncols_out
                k = 0
                for i in range(nrows_out):
                    for j in range(ncols_out):
                        for krow in range(self.kernel_size):
                            upd_col_shift = in_ch_shift+i*img_width+krow*img_width+self.stride*j
                            #print(f"row {out_ch_shift+k}, col {upd_col_shift}")
                            kernel_mat[out_ch_shift+k,upd_col_shift:upd_col_shift+self.kernel_size] = fltr[krow,:]
                        k+= 1
        output_tensor = kernel_mat.mm(input_tensor.reshape(nimages, input_tensor.numel()//nimages).transpose(0,1))
        output_tensor = output_tensor.transpose(0,1).reshape(nimages,self.out_channels, nrows_out, ncols_out)
        #print(f"got {output_tensor} of shape {output_tensor.shape}")
        #print(kernel_mat.shape)
        #print(kernel_mat[1])
        return output_tensor

class myConv2dMatmulV2(ABCConv2d):
    # Функция преобразования кернела в нужный формат.
    def _convert_kernel(self):
        converted_kernel = self.kernel.reshape(self.kernel.shape[0],-1)
        #print(converted_kernel)
        return converted_kernel

    # Функция преобразования входа в нужный формат.
    def _convert_input(self, input_tensor, output_height, output_width):
        sq_kernel =  self.kernel_size*self.kernel_size
        nrows = sq_kernel*self.in_channels
        ncols = output_height*output_width
        converted_input = torch.zeros( (nrows, input_tensor.shape[0]*ncols))
        #fill out the matrix
        for img_idx,img in enumerate(input_tensor):
            row_shift = 0
            col_shift = ncols*img_idx
            for in_ch_idx, in_ch in enumerate(img):
                #we only need to know how many total 
                #filter slides there will be to fill out the matrix
                k = 0
                for row in range(output_height):
                    for col in range(output_width):                        
                        tmp = in_ch[row*self.stride:row*self.stride+self.kernel_size,col*self.stride:col*self.stride+self.kernel_size]
                        converted_input[row_shift:row_shift+sq_kernel, col_shift+k:col_shift+k+1] = \
                            tmp.flatten().unsqueeze(1)
                        k += 1
                row_shift += sq_kernel 
        return converted_input

    def __call__(self, torch_input):
        batch_size, out_channels, output_height, output_width\
            = calc_out_shape(
                input_matrix_shape=torch_input.shape,
                out_channels=self.kernel.shape[0],
                kernel_size=self.kernel.shape[2],
                stride=self.stride,
                padding=0)

        converted_kernel = self._convert_kernel()
        converted_input = self._convert_input(torch_input, output_height, output_width)
        alt_out_mat = converted_kernel @ converted_input
       
        #rebuild original output matrix
        output_tensor = torch.zeros((torch_input.shape[0], self.out_channels, 
                                     output_height, output_width))
        
        for img in range(batch_size):
            for out_ch in range(out_channels):
                col_shift = img * output_height*output_width
                output_tensor[img][out_ch][:,:] = \
                    alt_out_mat[out_ch,col_shift:col_shift+output_height*output_width].reshape(output_height, output_width)
               
        return output_tensor

# функция, создающая объект класса cls и возвращающая свертку от input_matrix
def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
    out_channels = kernel.shape[0]
    in_channels = kernel.shape[1]
    kernel_size = kernel.shape[2]

    layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
    layer.set_kernel(kernel)

    return layer(input_matrix)


# Функция, тестирующая класс conv2d_cls.
# Возвращает True, если свертка совпадает со сверткой с помощью torch.nn.Conv2d.
def test_conv2d_layer(conv2d_layer_class, batch_size=2,
                      input_height=4, input_width=4, stride=2):
    kernel = torch.tensor(
                      [[[[0., 1, 0],
                         [1,  2, 1],
                         [0,  1, 0]],

                        [[1, 2, 1],
                         [0, 3, 3],
                         [0, 1, 10]],

                        [[10, 11, 12],
                         [13, 14, 15],
                         [16, 17, 18]]]])
    in_channels = kernel.shape[1]

    input_tensor = torch.arange(0, batch_size * in_channels *
                                input_height * input_width,
                                out=torch.FloatTensor()) \
        .reshape(batch_size, in_channels, input_height, input_width)

    custom_conv2d_out = create_and_call_conv2d_layer(
        conv2d_layer_class, stride, kernel, input_tensor)
    conv2d_out = create_and_call_conv2d_layer(
        Conv2d, stride, kernel, input_tensor)
       
    #print(f" proper output: {conv2d_out}, shape {conv2d_out.shape}")

    return torch.allclose(custom_conv2d_out, conv2d_out) \
              and (custom_conv2d_out.shape == conv2d_out.shape)

print(test_conv2d_layer(myConv2dMatmulV2, stride=1))
