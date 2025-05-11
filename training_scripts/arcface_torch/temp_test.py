import torch

x = torch.randn(5000, 5000).cuda()
for i in range(1000):
    x = x @ x
