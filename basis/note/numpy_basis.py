import torch
import numpy as np
# transform a torch tensor to a numpy array
a = torch.ones(5)
print("a:", a)
b = a.numpy()
print("b:", b)

# add 1 to a
a.add_(1)
"""tips: a and b share the same memory, so if you change a, b will change too"""
print("a:", a)
print("b:", b)

# transform a numpy array to a torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
print("a:", a)
print("b:", b)  # dtype=torch.float64

# add 1 to a
np.add(a, 1, out=a)
"""tips: a and b share the same memory, so if you change a, b will change too"""
print("a:", a)
print("b:", b)
