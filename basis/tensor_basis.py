# Load the PyTorch library
import torch

# create a empty tensor with shape (5,3)
x = torch.empty(5, 3)
print("torch.empty(5, 3):", x)

# create a tensor with random values with shape (5,3)
x = torch.rand(5, 3)
print("torch.rand(5, 3):", x)

# create a tensor with zeros values with shape (5,3),dtype=torch.long
x = torch.zeros(5, 3, dtype=torch.long)
print("torch.zeros(5, 3, dtype=torch.long):", x)

"""  tips:you can't create a tensor with random values with shape (5,3) and dtype=torch.long directly"""
# x = torch.rand(5, 3, dtype=torch.long)  # error
# replace:
x = torch.randn(5, 3)
x = x.to(torch.long)
print("torch.randn(5, 3).to(torch.long):", x)

# create a tensor with values [5.5,3]
x = torch.tensor([5.5, 3])
print("torch.tensor([5.5, 3]):", x)

# create a tensor with ones values with shape (5,3),dtype=torch.double
x = torch.ones(5, 3, dtype=torch.double)
print("torch.ones(5, 3, dtype=torch.double):", x)

# create a tensor with same shape and dtype with float tensor x
x = torch.randn_like(x, dtype=torch.float)
print("torch.randn_like(x, dtype=torch.float):", x)

# get the size of tensor x
print("x.size():", x.size())

# add two tensors
# syntax 1
y = torch.rand(5, 3)
print("x+y:", x + y)
# syntax 2
print("torch.add(x,y):", torch.add(x, y))
# syntax 3
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print("torch.add(x, y, out=result):", result)
# syntax 4
y.add_(x)
print("y.add_(x):", y)

# print the first line of tensor x
print("x[:,1]", x[:, 1])

# resize a 4x4 tensor to a 16 tensor
x = torch.randn(4, 4)
y = x.view(16)
print("x.view(16):", y)
print("x.size(),y.size():", x.size(), y.size())

# resize a 4x4 tensor to a 2x8 tensor
# syntax 1
y = x.view(2, 8)
print("x.view(2, 8):", y)
# syntax 2
z = x.view(-1, 8)
print("x.view(-1, 8):", z)

# get a number from tensor
x = torch.randn(1)
print("x.item():", x.item())
