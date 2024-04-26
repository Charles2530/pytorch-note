import torch
# create a tensor with ones values with required_grad=True
x = torch.ones(2, 2, requires_grad=True)
print("torch.ones(2, 2, requires_grad=True):", x)

# do any operation on tensor x
y = x + 2
print("x+2:", y)
print("y.grad_fn:", y.grad_fn)  # AddBackward Object is added to the graph

# do more operations on y
z = y * y * 3
out = z.mean()
print("z:", z)  # MulBackward Object is added to the graph
print("out:", out)  # MeanBackward Object is added to the graph

# do backward propagation on outss
out.backward()

# print the gradient of x
print("x.grad:", x.grad)

# create a tensor with result instance of the tensor
x = torch.randn(3, requires_grad=True)
y = x*2
while y.data.norm() < 1000:  # norm() is the L2 norm
    y = y*2

print("y:", y)

# calculate the gradient of y at v = [0.1, 1.0, 0.0001]
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print("x.grad:", x.grad)

# stop autograd from tracking history on tensors
# syntax 1
print("x.requires_grad:", x.requires_grad)
print("x.grad_fn:", x.grad_fn)
print("(x ** 2).requires_grad:", (x ** 2).requires_grad)

with torch.no_grad():
    print("(x ** 2).requires_grad:", (x ** 2).requires_grad)

# syntax 2
print("x.requires_grad:", x.requires_grad)
y = x.detach()
print("y.requires_grad:", y.requires_grad)
print("x.eq(y):", x.eq(y))
