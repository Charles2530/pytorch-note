from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
import os
import copy
from torch.nn import functional as F
import torch
import collections
import torch.nn as nn
# 几种定义模型的方式
# 1. nn.Sequential
net1 = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print(net1)
net2 = nn.Sequential(
    collections.OrderedDict([
        ('linear1', nn.Linear(784, 256)),
        ('relu', nn.ReLU()),
        ('linear2', nn.Linear(256, 10))
    ])
)  # ordered dict
print(net2)
a = torch.rand(4, 784)
out1 = net1(a)
out2 = net2(a)
print(out1.shape == out2.shape)
# 2. nn.ModuleList
net3 = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net3.append(nn.Linear(256, 10))
print(net3[-1])
print(net3)

# out3 = net3(a)  # 注意ModuleList没有实现forward方法,只是将不同的模块放在一起,所以需要进行如下使用


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
        self.linears.append(nn.Linear(256, 10))

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x


net3 = Net3()
out3 = net3(a)
print(out3.shape)


# 3. nn.ModuleDict
net4 = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU()
})
net4['output'] = nn.Linear(256, 10)
print(net4['linear'])
print(net4)
# out4 = net4(a)  # 同ModuleList,ModuleDict没有实现forward方法,所以需要进行如下使用


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.layers = nn.ModuleDict({
            'linear': nn.Linear(784, 256),
            'act': nn.ReLU()
        })
        self.layers['output'] = nn.Linear(256, 10)

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x


net4 = Net4()
out4 = net4(a)
print(out4.shape)
# 利用模块快速搭建模型(U-Net为例)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# 定义U-Net模型


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# 实例化模型
unet = UNet(n_channels=3, n_classes=1)
# print(unet)

# 模型修改
# 由于U-Net适用于二分类问题,所以输出通道数为1,但是在实际应用中,输出通道数可能不止一个,这时候需要对模型进行修改

# 添加额外输入
# 修改特定层
unet1 = copy.deepcopy(unet)
b = torch.rand(1, 3, 224, 224)
out_unet1 = unet1(b)
print(out_unet1.shape)
unet1.outc = OutConv(64, 5)
out_unet1 = unet1(b)
print(out_unet1.shape)


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, add_variable):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = x+add_variable  # add the extra variable
        logits = self.outc(x)
        return logits


unet2 = UNet2(n_channels=3, n_classes=1)
out_unet2 = unet2(b, torch.rand(1, 1, 224, 224))
print(out_unet2.shape)

# 添加额外输出


class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.outc2 = OutConv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, x5  # add the extra output


unet3 = UNet3(n_channels=3, n_classes=1)
out_unet3, mid_out = unet3(b)
print(out_unet3.shape)
print(mid_out.shape)

# 保存和加载模型
# print(unet.state_dict())
# 模型保存格式有三种pt, pth, pkl, 推荐使用pth
torch.save(unet.state_dict(
), '/home/charles/charles/python/pytorch/project/basis/note/model/unet.pth')
# 加载模型
# CPU或单卡:保存&读取整个模型
loaded_unet = torch.load(
    '/home/charles/charles/python/pytorch/project/basis/note/model/unet.pth')
# CPU或单卡:保存&读取模型权重
unet_weight = torch.load(
    '/home/charles/charles/python/pytorch/project/basis/note/model/unet.pth')
unet4 = UNet(n_channels=3, n_classes=1)
unet4.load_state_dict(unet_weight)
# 多卡:保存&读取整个模型,不建议使用
# 保存
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet_mul = copy.deepcopy(unet)
unet_mul = nn.DataParallel(unet_mul).to(device)
# 多卡:保存&读取模型权重
loaded_unet_weight = torch.load(
    '/home/charles/charles/python/pytorch/project/basis/note/model/unet.pth')
unet_mul.load_state_dict(loaded_unet_weight)
unet_mul = nn.DataParallel(unet_mul).to(device)

# 自定义损失函数


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


# 动态修改学习率
optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# 模型微调
# 1. 冻结模型的一部分参数
unet5 = UNet(n_channels=3, n_classes=1)
unet5.outc.conv.weight.requires_grad = False
# 将模型的参数设置为不可训练，以便微调
for layer, (name, param) in enumerate(unet5.named_parameters()):
    print(layer, name, param.requires_grad)
# 2. 修改优化器
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, unet5.parameters()), lr=0.001)
# 3. 模型微调
# unet5.train()

# 半精度训练
# 所谓半精度训练，即使用16位浮点数代替32位浮点数进行训练，这样可以减少显存占用，加快训练速度


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(
            '.jpg', '_mask.gif'))
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        return image, mask


def train_with_amp(model, optimizer, criterion, dataloader, scaler):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss
