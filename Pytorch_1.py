import pandas as pd
import numpy as np
import torch

data = [[1, 2, 3],[3, 4, 5]]
np_data = np.array(data)
x_data = torch.tensor(data)

print(x_data)
print(torch.tensor(np_data))

# _like will maintain the input data shape and types
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


#shape is used as a template for torch
shape = (6,6)
ones = torch.ones(shape)
zeros = torch.zeros(shape)
rando = torch.randn(shape)
print(ones)
print(zeros)
print(rando)
