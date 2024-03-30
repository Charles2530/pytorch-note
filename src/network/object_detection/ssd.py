import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
# SSD（单发多框检测）是另一种流行的目标检测算法
# SSD使用多个锚框来检测目标
# SSD使用一个卷积神经网络来提取特征
# SSD使用一个卷积层来预测每个锚框的类别和偏移量


# SSD模型
# 1. 使用一个预训练的模型来提取特征
# 2. 使用一个卷积层来预测每个锚框的类别和偏移量
# 3. 使用一个RoI池化层来提取特征
# 4. 使用一个全连接层来分类
# 5. 使用另一个全连接层来预测边界框
# 6. 使用非极大值抑制来移除重叠的预测边界框

# 课设实现
img = d2l.plt.imread('./img/catdog.jpg')
h, w = img.shape[0:2]
# display_anchors作用是显示锚框


def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    fmap = torch.zeros((1, 10, fmap_h, fmap_w), dtype=torch.float32)
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
    plt.show()


display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
