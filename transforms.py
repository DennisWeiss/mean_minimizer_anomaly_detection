import random

import torch
import torchvision
from PIL import Image, ImageOps, ImageFilter


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def get_color_distortion(scale=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = torchvision.transforms.ColorJitter(0.8 * scale, 0.8 * scale, 0.8 * scale, 0.2 * scale)
    rnd_color_jitter = torchvision.transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = torchvision.transforms.RandomGrayscale(p=0.2)
    color_distort = torchvision.transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class Transform:
    def __init__(self):
        transform1 = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(32),
                                          torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                          get_color_distortion(scale=0.5),
                                          torchvision.transforms.ToTensor()])

        self.transforms = [torchvision.transforms.Identity(), transform1]

    def __call__(self, x):
        return (transform(x) for transform in self.transforms)

