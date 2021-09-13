#! /usr/bin/env python3

from numpy.core.numeric import convolve
import torch
import numpy as np
from torch._C import dtype
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# 10 batches, each vector of 2 elements
x = np.arange((20), dtype=np.double)
x = np.reshape(x, (10,2))
x = torch.from_numpy(x)


lin = nn.Linear(2,10)

print(lin.weight.data)
print(lin.bias.data)
