#! /usr/bin/env python3

from numpy.core.numeric import convolve
import torch
import numpy as np
from torch._C import dtype
import torch.nn as nn
from functions import ReverseLayerF

torch.set_default_dtype(torch.float64)

batch_size = 1
num_samples = 20
num_channels = 5

channels = []
for i in range(1,1+num_channels):
    channels.append(
        np.ones(shape=(batch_size, num_samples)) * i
    )

x = np.stack(channels, axis=1)
x = torch.from_numpy(x)
print(x.shape)


##########################################################################################################
depthwise_conv1d = nn.Conv1d(in_channels=num_channels, out_channels=10, kernel_size=10, stride=1, groups=num_channels)
# Each channel gets its own (out_channels / groups) number of filters. These are applied, and then simply stacked in the output
depthwise_conv1d.weight.data.fill_(1)
depthwise_conv1d.bias.data.fill_(0.)
convolved = depthwise_conv1d(x)
##########################################################################################################

# print(x.shape)
# x = x.permute(1,0,2)
# print(x.shape)
# I = x[0]
# Q = x[1]

# I = I.reshape((batch_size, 1, num_samples))
# Q = Q.reshape((batch_size, 1, num_samples))


# conv1d_I = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=10, stride=1)
# conv1d_Q = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=10, stride=1)

# conv1d_I.weight.data.fill_(1)
# conv1d_I.bias.data.fill_(0.)
# conv1d_Q.weight.data.fill_(1)
# conv1d_Q.bias.data.fill_(0.)

# convolved_I = conv1d_I(I)
# convolved_Q = conv1d_Q(Q)

# back = torch.stack((convolved_I, convolved_Q))
# back = back.permute(1,0,2,3)

# print(back.shape)
# print(back[0])

print(convolved.shape)

print(convolved)
