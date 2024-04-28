import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torchinfo import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet18 = models.resnet18().to(device)
# 使用torchinfo查看模型的结构
print(summary(resnet18, input_size=(1, 3, 224, 224)))

# 可视化CNN的卷积核
print(dict(resnet18.named_children()))
print(dict(resnet18.named_children())['conv1'])
conv1 = dict(resnet18.named_children())['conv1']
kernel_set = conv1.weight.cpu().detach()
num = len(kernel_set)
print(num, kernel_set.shape)
for i in range(0, num):
    i_kernel = kernel_set[i]
    plt.figure(figsize=(20, 17))
    if (len(i_kernel) > 1):
        for idx, filter in enumerate(i_kernel):
            plt.subplot(9, 9, idx+1)
            plt.axis('off')
            plt.imshow(filter[:, :].detach(), cmap='gray')
# plt.show()

# 可视化CNN的特征图


class Hook(object):
    def __init__(self):
        self.module_name = []
        self.features_in_hook = []
        self.features_out_hook = []

    def __call__(self, module, fea_in, fea_out):
        self.module_name.append(module.__class__)
        self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)
        return None


inputs = torch.randn(1, 3, 224, 224).to(device)
hk = Hook()
resnet18.conv1.register_forward_hook(hk)
