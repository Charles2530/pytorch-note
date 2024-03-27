import torch
A = torch.tensor([[1.0, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
x = torch.tensor([1.0, 2, 3, 4, 5])
y = torch.mv(A, x)
print(y)
B = torch.tensor([[1.0, 2, 3], [4, 5, 6], [7, 8, 9],
                 [10, 11, 12], [13, 14, 15]])
z = torch.mm(A, B)
print(z)
print(torch.norm(x))
print(torch.norm(torch.ones(5)))
