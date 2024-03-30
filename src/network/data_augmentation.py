import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from d2l import torch as d2l
# 数据增强通过对数据变形来获取多样性，从而提高模型的泛化能力
# 数据增强的方式有很多种，比如旋转、翻转、裁剪、缩放、平移、仿射变换、颜色变换等


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.hflip(sample['image'])
            sample['mask'] = F.hflip(sample['mask'])
        return sample


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.vflip(sample['image'])
            sample['mask'] = F.vflip(sample['mask'])
        return sample


class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, sample):
        angle = transforms.RandomRotation.get_params(self.degrees)
        sample['image'] = F.rotate(
            sample['image'], angle, self.resample, self.expand, self.center)
        sample['mask'] = F.rotate(
            sample['mask'], angle, self.resample, self.expand, self.center)
        return sample


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=InterpolationMode.BILINEAR):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, sample):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            sample['image'], scale=self.scale, ratio=self.ratio)
        sample['image'] = F.resized_crop(
            sample['image'], i, j, h, w, self.size, self.interpolation)
        sample['mask'] = F.resized_crop(
            sample['mask'], i, j, h, w, self.size, self.interpolation)
        return sample


class RandomAffine(object):
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, sample):
        angle = transforms.RandomAffine.get_params(
            self.degrees, self.translate, self.scale, self.shear, sample['image'].size)
        sample['image'] = F.affine(
            sample['image'], *angle, resample=self.resample, fillcolor=self.fillcolor)
        sample['mask'] = F.affine(
            sample['mask'], *angle, resample=self.resample, fillcolor=self.fillcolor)
        return sample


class RandomColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        sample['image'] = F.adjust_brightness(sample['image'], self.brightness)
        sample['image'] = F.adjust_contrast(sample['image'], self.contrast)
        sample['image'] = F.adjust_saturation(sample['image'], self.saturation)
        sample['image'] = F.adjust_hue(sample['image'], self.hue)
        return sample


class RandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.to_grayscale(
                sample['image'], num_output_channels=3)
        return sample


class RandomPerspective(object):
    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR):
        self.distortion_scale = distortion_scale
        self.p = p
        self.interpolation = interpolation

    def __call__(self, sample):
        if random.random() < self.p:
            width, height = sample['image'].size
            startpoints = torch.tensor([[0, 0], [0, height], [width, 0], [
                                       width, height]], dtype=torch.float32)
            endpoints = startpoints + torch.randn(4, 2) * self.distortion_scale
            sample['image'] = F.perspective(
                sample['image'], startpoints, endpoints, self.interpolation)
            sample['mask'] = F.perspective(
                sample['mask'], startpoints, endpoints, self.interpolation)
        return sample


class RandomErasing(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.random_erasing(
                sample['image'], self.scale, self.ratio, self.value, self.inplace)
        return sample


class RandomGaussianBlur(object):
    def __init__(self, p=0.5, kernel_size=3):
        self.p = p
        self.kernel_size = kernel_size

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.gaussian_blur(
                sample['image'], self.kernel_size)
        return sample


class RandomSolarize(object):
    def __init__(self, p=0.5, threshold=128):
        self.p = p
        self.threshold = threshold

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.solarize(sample['image'], self.threshold)
        return sample


class RandomPosterize(object):
    def __init__(self, p=0.5, bits=4):
        self.p = p
        self.bits = bits

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.posterize(sample['image'], self.bits)
        return sample


class RandomSharpness(object):
    def __init__(self, p=0.5, sharpness=2.0):
        self.p = p
        self.sharpness = sharpness

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.adjust_sharpness(
                sample['image'], self.sharpness)
        return sample


class RandomEqualize(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.equalize(sample['image'])
        return sample


class RandomInvert(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.invert(sample['image'])
        return sample


class RandomAutocontrast(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.autocontrast(sample['image'])
        return sample


class RandomAdjustSharpness(object):
    def __init__(self, p=0.5, sharpness_factor=2.0):
        self.p = p
        self.sharpness_factor = sharpness_factor

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.adjust_sharpness(
                sample['image'], self.sharpness_factor)
        return sample


class RandomAdjustContrast(object):
    def __init__(self, p=0.5, contrast_factor=2.0):
        self.p = p
        self.contrast_factor = contrast_factor

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.adjust_contrast(
                sample['image'], self.contrast_factor)
        return sample


class RandomAdjustBrightness(object):
    def __init__(self, p=0.5, brightness_factor=1.0):
        self.p = p
        self.brightness_factor = brightness_factor

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.adjust_brightness(
                sample['image'], self.brightness_factor)
        return sample


class RandomAdjustSaturation(object):
    def __init__(self, p=0.5, saturation_factor=1.0):
        self.p = p
        self.saturation_factor = saturation_factor

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.adjust_saturation(
                sample['image'], self.saturation_factor)
        return sample


class RandomAdjustHue(object):
    def __init__(self, p=0.5, hue_factor=0.0):
        self.p = p
        self.hue_factor = hue_factor

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.adjust_hue(sample['image'], self.hue_factor)
        return sample


class RandomAdjustGamma(object):
    def __init__(self, p=0.5, gamma=2.0):
        self.p = p
        self.gamma = gamma

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.adjust_gamma(sample['image'], self.gamma)
        return sample


class RandomAdjustPosterize(object):
    def __init__(self, p=0.5, bits=4):
        self.p = p
        self.bits = bits

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.posterize(sample['image'], self.bits)
        return sample


class RandomAdjustSharpness(object):
    def __init__(self, p=0.5, sharpness_factor=2.0):
        self.p = p
        self.sharpness_factor = sharpness_factor

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.adjust_sharpness(
                sample['image'], self.sharpness_factor)
        return sample


class RandomAdjustContrast(object):
    def __init__(self, p=0.5, contrast_factor=2.0):
        self.p = p
        self.contrast_factor = contrast_factor

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.adjust_contrast(
                sample['image'], self.contrast_factor)
        return sample


class RandomAdjustBrightness(object):
    def __init__(self, p=0.5, brightness_factor=1.0):
        self.p = p
        self.brightness_factor = brightness_factor

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.adjust_brightness(
                sample['image'], self.brightness_factor)
        return sample


# 下面为课设代码示例
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img)


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)


apply(img, RandomHorizontalFlip(), num_rows=1, num_cols=4)
