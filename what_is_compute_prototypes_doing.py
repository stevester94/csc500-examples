#! /usr/bin/env python3
# import torch
# import unittest
# import numpy as np

# from steves_utils.ORACLE.torch_utils import ORACLE_Torch_Dataset
# from steves_utils.ORACLE.torch_utils import build_ORACLE_episodic_iterable
# from steves_utils.ORACLE.utils_v2 import (
#     ALL_DISTANCES_FEET,
#     ALL_SERIAL_NUMBERS,
#     ALL_RUNS,
#     serial_number_to_id
# )

# num_examples_per_device = 7500
# n_train_tasks = 200
# n_val_tasks = 100
# n_test_tasks = 10
# max_cache_items = int(4.5e6)
# desired_serial_numbers=ALL_SERIAL_NUMBERS[:5]
# desired_distances=ALL_DISTANCES_FEET[:5]
# desired_runs=[1]
# n_way=len(ALL_SERIAL_NUMBERS[:5])
# n_shot=2
# n_query=2
# window_length=256
# window_stride=50
# seed=1337


# n_train_tasks_per_distance=int(n_train_tasks/len(desired_distances))
# n_val_tasks_per_distance=int(n_val_tasks/len(desired_distances))
# n_test_tasks_per_distance=int(n_test_tasks/len(desired_distances))
# max_cache_size_per_distance=int(max_cache_items/len(desired_distances))
# num_examples_per_device_per_distance=int(num_examples_per_device/len(desired_distances))

# train_dl, val_dl, test_dl = build_ORACLE_episodic_iterable(
#     desired_serial_numbers=desired_serial_numbers,
#     # desired_distances=[50],
#     desired_distances=desired_distances,
#     desired_runs=desired_runs,
#     window_length=window_length,
#     window_stride=window_stride,
#     num_examples_per_device_per_distance=num_examples_per_device_per_distance,
#     seed=seed,
#     max_cache_size_per_distance=0,
#     # n_way=len(ALL_SERIAL_NUMBERS),
#     n_way=n_way,
#     n_shot=n_shot,
#     n_query=n_query,
#     n_train_tasks_per_distance=n_train_tasks_per_distance,
#     n_val_tasks_per_distance=n_val_tasks_per_distance,
#     n_test_tasks_per_distance=n_test_tasks_per_distance,
# )


# for k in train_dl:
#     print(k[1][1], k[1][3])



# import torch
# import numpy as np
# def compute_prototypes(
#     support_features: torch.Tensor, support_labels: torch.Tensor
# ) -> torch.Tensor:
#     """
#     Compute class prototypes from support features and labels
#     Args:
#         support_features: for each instance in the support set, its feature vector
#         support_labels: for each instance in the support set, its label

#     Returns:
#         for each label of the support set, the average feature vector of instances with this label
#     """

#     n_way = len(torch.unique(support_labels))
#     print("n_way", n_way)
#     # Prototype i is the mean of all instances of features corresponding to labels == i

#     for label in range(n_way):
#         print(
#             torch.nonzero(support_labels == label)
#         )

#     return torch.cat(
#         [
#             support_features[torch.nonzero(support_labels == label)].mean(0)
#             for label in range(n_way)
#         ]
#     )

# support = [
#     (torch.from_numpy(np.array([1,1], dtype=np.float)), 1)
#     (torch.from_numpy(np.array([1,1], dtype=np.float)), 1)
#     (torch.from_numpy(np.array([2,2], dtype=np.float)), 2)
#     (torch.from_numpy(np.array([2,2], dtype=np.float)), 2)
#     (torch.from_numpy(np.array([3,3], dtype=np.float)), 3)
#     (torch.from_numpy(np.array([3,3], dtype=np.float)), 3)
# ]

# query = [
#     (torch.from_numpy(np.array([1,1], dtype=np.float)), 1)
#     (torch.from_numpy(np.array([2,2], dtype=np.float)), 2)
#     (torch.from_numpy(np.array([3,3], dtype=np.float)), 3)
# ]




# features = [
#     [1,1],[1,1],
#     [2,2],[2,2],
#     [3,3],[3,3],
# ]

# labels = [
#     0,0,
#     1,1,
#     2,2,
# ]

# features = np.array(features, dtype=np.float)
# labels   = np.array(labels, dtype=np.float)

# features = torch.from_numpy(features)
# labels = torch.from_numpy(labels)

# ret = compute_prototypes(features, labels)

# print(ret)


# loss = torch.nn.CrossEntropyLoss()

import torch
import numpy as np

input = torch.from_numpy(np.array([
    [9] + [0.0]*15,
    [9] + [0.0]*15,
    [9] + [0.0]*15,
], dtype=float))

target = torch.from_numpy(np.array([
    15,
    15,
    15,
]))

input = torch.nn.LogSoftmax()(input)

# loss(scores, y)
loss = torch.nn.CrossEntropyLoss()
loss = torch.nn.NLLLoss()
# input = torch.randn(1, 5, requires_grad=True)
# target = torch.empty(1, dtype=torch.long).random_(5)

print(input)
print(target)

output = loss(input, target)
print(output)
