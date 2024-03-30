import torch
from torch import nn
from d2l import torch as d2l
import torchvision
# 区域卷积神经网络（R-CNN）
# R-CNN（区域卷积神经网络）是一个经典的目标检测算法。
# 使用启发式搜索算法来选择锚框，使用预训练的卷积神经网络来提取特征
# 训练一个SVM来对类别分类
# 训练一个线性回归模型来预测边界框偏移

# 兴趣区域池化层(RoI)：将不同大小的兴趣区域池化到相同大小的特征图上
# 给定一个锚框，均匀分割为nxm块，输出每块的最大值
# 不管输入的大小，输出的大小是固定的

# Fast R-CNN：使用CNN提取特征，然后使用RoI池化层对每个锚框生成固定大小的特征图
# Mask R-CNN：在Fast R-CNN的基础上增加了一个分支，用于预测每个像素的类别
# Faster R-CNN：使用RPN（区域提议网络）来生成锚框，然后使用RoI池化层来提取特征
# Faster R-CNN和Mask R-CNN是目前最先进的目标检测算法,在高精度场景下常用算法


def roi_pooling(X, rois, output_size, xy_indices):
    # X: 输入特征图
    # rois: 包含n个锚框的张量
    # output_size: 输出的大小
    # xy_indices: 锚框的索引
    Y = []
    for i in range(rois.shape[0]):
        roi = rois[i]
        x_indices = xy_indices[i][:, 0].long()
        y_indices = xy_indices[i][:, 1].long()
        Y.append(nn.functional.adaptive_max_pool2d(
            X[:, :, y_indices, x_indices], output_size))
    return torch.cat(Y, dim=0)

# 为了演示R-CNN，我们使用一个简化的Faster R-CNN模型
# 1. 使用一个预训练的模型来提取特征
# 2. 使用一个全连接层来分类
# 3. 使用另一个全连接层来预测边界框


def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


faster_rcnn = nn.Sequential()
# 使用一个预训练的模型来提取特征
features = torchvision.models.resnet18(pretrained=True)
features = nn.Sequential(*list(features.children())[:-2])
faster_rcnn.add_module('features', features)
# 使用一个全连接层来分类
faster_rcnn.add_module('classifier', nn.Sequential(
    nn.Linear(512, 21)))
# 使用另一个全连接层来预测边界框
faster_rcnn.add_module('bounding_box', nn.Linear(512, 84))
