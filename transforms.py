import random

import torch
import torchvision
import torchvision.transforms.functional as F
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


def get_color_distortion(scale=0.5):
    color_jitter = torchvision.transforms.ColorJitter(0.8 * scale, 0.8 * scale, 0.8 * scale, 0.2 * scale)
    return color_jitter


class Transform:
    def __init__(self, test=False):
        transform1 = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(32),
                                                     torchvision.transforms.ToTensor()])

        transform2 = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=3),
                                                     torchvision.transforms.ToTensor()])

        transform3 = torchvision.transforms.Compose([get_color_distortion(scale=0.5),
                                                     torchvision.transforms.ToTensor()])

        total_transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(32),
                                                          torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                          get_color_distortion(scale=0.5),
                                                          torchvision.transforms.ToTensor()])

        transform_x = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(32, scale=(0.08, 0.8)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            get_color_distortion(scale=0.5),
            torchvision.transforms.ToTensor()]
        )

        # self.transforms = [torchvision.transforms.ToTensor() for i in range(4)]

        if False:
            self.transforms = [torchvision.transforms.ToTensor() for i in range(4)]
        else:
            self.transforms = [torchvision.transforms.ToTensor(), torchvision.transforms.ToTensor(), torchvision.transforms.ToTensor(), torchvision.transforms.ToTensor()]

    def __call__(self, x):
        return [transform(x) for transform in self.transforms]

