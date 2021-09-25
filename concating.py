#! /usr/bin/env python3

from numpy.core.numeric import convolve
import torch
import numpy as np
from torch._C import dtype
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# 10 batches, each vector of 2 elements
x = np.zeros((10,2))
x = torch.from_numpy(x)

y = np.ones((10))
y = torch.from_numpy(y)

y = y.reshape(-1,1)

print(x)
print(y)

c = torch.cat([x,y], -1)
print(c)

