#! /usr/bin/env python3

import numpy as np
import torch


pred = [
	[0,0,0,1],
]

truth = [
	3
]

pred=torch.from_numpy(np.array(pred, dtype=np.single))
truth=torch.from_numpy(np.array(truth, dtype=np.int))


loss = torch.nn.NLLLoss()

print(loss(pred, truth))