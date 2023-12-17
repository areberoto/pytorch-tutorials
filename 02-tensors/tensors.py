"""
Copyright 2023 PyTorch
SPDX-License-Identifier: BSD-3-Clause

Tensors tutorial, basic of tensors
"""

import torch
import numpy as np

# Initialize tensor directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"Tensor from data: \n {x_data} \n")

# Create tensor from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from tensor: \n {x_np} \n")

# Create tensor from another tensor, retaining properties(shape, datatype, etc.)
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# shape is a tuple of tensor dimensions:
shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# Attributes of a tensor:
tensor = torch.rand(3, 4)
print(f"Shape of Tensor: \n {tensor.shape} \n")
print(f"Datatype of Tensor: \n {tensor.dtype} \n")
print(f"Device tensor is stored on: \n {tensor.device} \n")

# By default, are crated on CPU. If GPU available, move to GPU:
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

print(f"Device tensor is stored on: \n {tensor.device} \n")

# Tensor operations:
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

# Joining tensors:
t1 = torch.cat([tensor, tensor, tensor, tensor], dim=1)
print(t1)

t1 = torch.cat([tensor, tensor, tensor, tensor], dim=0)
print(t1)

# Arithmetic operations:

# Compute matrix multiplication. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

print(y3)

# Compute element-wise product. z1, z2, z3 will have the same value:
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(z3)

# Single-element tensors: convert one-element tensor to Python numerical value:
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations: Denoted by _ suffix, store the result into the operand:
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Bridge with NumPy. Tensor to numpy array, sharing their underlying memory locations:
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {t}")

# A change in the tensor reflects in the numpy array:
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Changes in the numpy array reflects in the tensor:
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
