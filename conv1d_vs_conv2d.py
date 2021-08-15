#! /usr/bin/env python3

from numpy.core.numeric import convolve
import torch
import numpy as np
from torch._C import dtype
import torch.nn as nn
from functions import ReverseLayerF

torch.set_default_dtype(torch.float64)


x = np.arange((15), dtype=np.double)
x = np.reshape(x, (1,3,5))
x = torch.from_numpy(x)


conv1d = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, stride=1)
conv1d.weight.data.fill_(1)
conv1d.bias.data.fill_(0.)
y = conv1d(x)
print("========================================================================================")
print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Input:", x)
print("Output:", y)
print("========================================================================================")


conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), stride=1)
conv2d.weight.data.fill_(1)
conv2d.bias.data.fill_(0.)
x = x.reshape((1,1,3,5))
y = conv2d(x)
print("========================================================================================")
print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Input:", x)
print("Output:", y)
print("========================================================================================")