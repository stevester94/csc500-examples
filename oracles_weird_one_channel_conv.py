#! /usr/bin/env python3

from numpy.core.numeric import convolve
import torch
import numpy as np
from torch._C import dtype
import torch.nn as nn
from functions import ReverseLayerF

torch.set_default_dtype(torch.float64)

batch_size = 200
num_samples = 128

# I dimension is ones, Q dimension is zeros
x_I = np.ones(shape=(batch_size, num_samples))
x_Q = np.zeros(shape=(batch_size, num_samples))
x = np.stack((x_I, x_Q), axis=1)
x = torch.from_numpy(x)

##########################################################################################################
depthwise_conv1d = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=10, stride=1, groups=2)
# So idk how this shit generalizes, but each channel gets its own filters, and is responsible for <groups> filters. The output is just concatenated.
# Yeah idk but this is on the right track, the output tells the whole story
depthwise_conv1d.weight.data.fill_(1)
depthwise_conv1d.bias.data.fill_(0.)
convolved = depthwise_conv1d(x)
##########################################################################################################

print(x.shape)
x = x.permute(1,0,2)
print(x.shape)
I = x[0]
Q = x[1]

I = I.reshape((batch_size, 1, num_samples))
Q = Q.reshape((batch_size, 1, num_samples))


conv1d_I = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=10, stride=1)
conv1d_Q = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=10, stride=1)

conv1d_I.weight.data.fill_(1)
conv1d_I.bias.data.fill_(0.)
conv1d_Q.weight.data.fill_(1)
conv1d_Q.bias.data.fill_(0.)

convolved_I = conv1d_I(I)
convolved_Q = conv1d_Q(Q)

back = torch.stack((convolved_I, convolved_Q))
back = back.permute(1,0,2,3)

print(back.shape)
print(convolved.shape)

print(convolved[0])
print(back[0])